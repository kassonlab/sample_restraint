//
// Created by Eric Irrgang on 2/26/18.
//

#include "ensemblepotential.h"

#include "gmxapi/session.h"
#include "gmxapi/md/mdsignals.h"

#include <cmath>

#include <vector>

namespace plugin
{

// Explicit instantiation.
template class ::plugin::Matrix<double>;

void EnsembleResourceHandle::reduce(const Matrix<double> &send,
                                    Matrix<double> *receive) const
{
    assert(reduce_);
    // Should probably check that the function object has been initialized...
    (*reduce_)(send, receive);
}

void EnsembleResourceHandle::stop()
{
    auto signaller = gmxapi::getMdrunnerSignal(session_, gmxapi::md::signals::STOP);

    // Should probably check that the function object has been initialized...
    signaller();
}

gmxapi::context::OutputStream *EnsembleResourceHandle::ostream()
{
    return ostream_.get();
}

/*!
 * \brief Apply a Gaussian blur when building a density grid for a list of values.
 *
 * Normalize such that the area under each sample is 1.0/num_samples.
 */
class BlurToGrid
{
    public:
        /*!
         * \brief Contsruct the blurring functor.
         *
         * \param low The coordinate value of the first grid point.
         * \param gridSpacing Distance between grid points.
         * \param sigma Gaussian parameter for blurring inputs onto the grid.
         */
        BlurToGrid(double low,
                   double gridSpacing,
                   double sigma) :
            low_{low},
            binWidth_{gridSpacing},
            sigma_{sigma}
        {
        };

        /*!
         * \brief Callable for the functor.
         *
         * \param samples A list of values to be blurred onto the grid.
         * \param grid Pointer to the container into which to accumulate a blurred histogram of samples.
         *
         * Example:
         *
         *     # Acquire 3 samples to be discretized with blurring.
         *     std::vector<double> someData = {3.7, 8.1, 4.2};
         *
         *     # Create an empty grid to store magnitudes for points 0.5, 1.0, ..., 10.0.
         *     std::vector<double> histogram(20, 0.);
         *
         *     # Specify the above grid and a Gaussian parameter of 0.8.
         *     auto blur = BlurToGrid(0.5, 0.5, 0.8);
         *
         *     # Collect the density grid for the samples.
         *     blur(someData, &histogram);
         *
         */
        void operator() (const std::vector<double>& samples, std::vector<double>* grid)
        {
            const auto nbins = grid->size();
            const double& dx{binWidth_};
            const auto num_samples = samples.size();

            const double denominator = 1.0/(2*sigma_*sigma_);
            const double normalization = 1.0/(num_samples*sqrt(2.0*M_PI*sigma_*sigma_));
            // We aren't doing any filtering of values too far away to contribute meaningfully, which
            // is admittedly wasteful for large sigma...
            for (size_t i = 0; i < nbins; ++i)
            {
                double bin_value{0};
                const double bin_x{low_ + i*dx};
                for(const auto distance : samples)
                {
                    const double relative_distance{bin_x - distance};
                    const auto numerator = -relative_distance*relative_distance;
                    bin_value += normalization*exp(numerator*denominator);
                }
                grid->at(i) = bin_value;
            }
        };

    private:
        /// Minimum value of bin zero
        const double low_;

        /// Size of each bin
        const double binWidth_;

        /// Smoothing factor
        const double sigma_;
};

EnsembleHarmonic::EnsembleHarmonic(size_t nbins,
                                   double binWidth,
                                   double minDist,
                                   double maxDist,
                                   PairHist experimental,
                                   unsigned int nSamples,
                                   double samplePeriod,
                                   unsigned int nWindows,
                                   double k,
                                   double sigma) :
    nBins_{nbins},
    binWidth_{binWidth},
    minDist_{minDist},
    maxDist_{maxDist},
    histogram_(nbins, 0),
    experimental_{std::move(experimental)},
    nSamples_{nSamples},
    currentSample_{0},
    samplePeriod_{samplePeriod},
    // In actuality, we have nsamples at (samplePeriod - dt), but we don't have access to dt.
    nextSampleTime_{samplePeriod},
    distanceSamples_(nSamples),
    nWindows_{nWindows},
    currentWindow_{0},
    windowStartTime_{0},
    nextWindowUpdateTime_{nSamples*samplePeriod},
    windows_{},
    k_{k},
    sigma_{sigma}
{}

EnsembleHarmonic::EnsembleHarmonic(const input_param_type &params) :
    EnsembleHarmonic(params.nBins,
                     params.binWidth,
                     params.minDist,
                     params.maxDist,
                     params.experimental,
                     params.nSamples,
                     params.samplePeriod,
                     params.nWindows,
                     params.k,
                     params.sigma)
{
}

// Todo: reference coordinate for PBC problems.
void EnsembleHarmonic::callback(gmx::Vector v,
                                gmx::Vector v0,
                                double t,
                                const EnsembleResources &resources)
{
    auto rdiff = v - v0;
    const auto Rsquared = dot(rdiff,
                              rdiff);
    const auto R = sqrt(Rsquared);

    // Store historical data every sample_period steps
    if (t >= nextSampleTime_)
    {
        distanceSamples_[currentSample_++] = R;
        nextSampleTime_ = (currentSample_ + 1)*samplePeriod_ + windowStartTime_;
    };

    // Every nsteps:
    //   0. Drop oldest window
    //   1. Reduce historical data for this restraint in this simulation.
    //   2. Call out to the global reduction for this window.
    //   3. On update, checkpoint the historical data source.
    //   4. Update historic windows.
    //   5. Use handles retained from previous windows to reconstruct the smoothed working histogram
    if (t >= nextWindowUpdateTime_)
    {
        // Get next histogram array, recycling old one if available.
        std::unique_ptr<Matrix<double>> new_window = gmx::compat::make_unique<Matrix<double>>(1,
                                                                                              nBins_);
        std::unique_ptr<Matrix<double>> temp_window;
        if (windows_.size() == nWindows_)
        {
            // Recycle the oldest window.
            // \todo wrap this in a helper class that manages a buffer we can shuffle through.
            windows_[0].swap(temp_window);
            windows_.erase(windows_.begin());
        }
        else
        {
            auto new_temp_window = gmx::compat::make_unique<Matrix<double>>(1,
                                                                            nBins_);
            assert(new_temp_window);
            temp_window.swap(new_temp_window);
        }

        // Reduce sampled data for this restraint in this simulation, applying a Gaussian blur to fill a grid.
        auto blur = BlurToGrid(0.0,
                               binWidth_,
                               sigma_);
        assert(new_window != nullptr);
        assert(distanceSamples_.size() == nSamples_);
        assert(currentSample_ == nSamples_);
        blur(distanceSamples_,
             new_window->vector());
        // We can just do the blur locally since there aren't many bins. Bundling these operations for
        // all restraints could give us a chance at some parallelism. We should at least use some
        // threading if we can.

        // We request a handle each time before using resources to make error handling easier if there is a failure in
        // one of the ensemble member processes and to give more freedom to how resources are managed from step to step.
        auto ensemble = resources.getHandle();
        // Get global reduction (sum) and checkpoint.
        assert(temp_window != nullptr);
        // Todo: in reduce function, give us a mean instead of a sum.
        ensemble.reduce(*new_window,
                        temp_window.get());

        // Update window list with smoothed data.
        windows_.emplace_back(std::move(new_window));

        // Get new histogram difference. Subtract the experimental distribution to get the values to use in our potential.
        for (auto &bin : histogram_)
        {
            bin = 0;
        }
        for (const auto &window : windows_)
        {
            for (size_t i = 0; i < window->cols(); ++i)
            {
                histogram_.at(i) += (window->vector()->at(i) - experimental_.at(i))/windows_.size();
            }
        }


        // Note we do not have the integer timestep available here. Therefore, we can't guarantee that updates occur
        // with the same number of MD steps in each interval, and the interval will effectively lose digits as the
        // simulation progresses, so _update_period should be cleanly representable in binary. When we extract this
        // to a facility, we can look for a part of the code with access to the current timestep.
        windowStartTime_ = t;
        nextWindowUpdateTime_ = nSamples_*samplePeriod_ + windowStartTime_;
        ++currentWindow_; // This is currently never used. I'm not sure it will be, either...

        // Reset sample bufering.
        currentSample_ = 0;
        // Reset sample times.
        nextSampleTime_ = t + samplePeriod_;
    };

}

gmx::PotentialPointData EnsembleHarmonic::calculate(gmx::Vector v,
                                                    gmx::Vector v0,
                                                    double t)
{
    // This is not the vector from v to v0. It is the position of a site
    // at v, relative to the origin v0. This is a potentially confusing convention...
    auto rdiff = v - v0;
    const auto Rsquared = dot(rdiff,
                              rdiff);
    const auto R = sqrt(Rsquared);


    // Compute output
    gmx::PotentialPointData output;
    // Energy not needed right now.
//    output.energy = 0;

    if (R != 0) // Direction of force is ill-defined when v == v0
    {

        double f{0};

        if (R > maxDist_)
        {
            // apply a force to reduce R
            f = k_ * (maxDist_ - R);
        }
        else if (R < minDist_)
        {
            // apply a force to increase R
            f = k_ * (minDist_ - R);
        }
        else
        {
            double f_scal{0};

            const size_t numBins = histogram_.size();
            double normConst = sqrt(2*M_PI)*sigma_*sigma_*sigma_;

            for (size_t n = 0; n < numBins; n++)
            {
                const double x{n*binWidth_ - R};
                const double argExp{-0.5*x*x/(sigma_*sigma_)};
                f_scal += histogram_.at(n)*exp(argExp)*x/normConst;
            }
            f = -k_ * f_scal;
        }

        output.force = f / norm(rdiff) * rdiff;
    }
    return output;
}

EnsembleResourceHandle EnsembleResources::getHandle() const
{
    auto handle = EnsembleResourceHandle();

    assert(bool(reduce_));
    handle.reduce_ = &reduce_;

    assert(session_);
    handle.session_ = session_;

    return handle;
}

void EnsembleResources::setSession(gmxapi::Session* session)
{
    assert(session);
    session_ = session;
}

void EnsembleResources::setOutputStream(std::unique_ptr<gmxapi::context::OutputStream> ostream)
{
    ostream_ = std::move(ostream);
}

// Explicitly instantiate a definition.
template class ::plugin::RestraintModule<EnsembleRestraint>;

} // end namespace plugin
