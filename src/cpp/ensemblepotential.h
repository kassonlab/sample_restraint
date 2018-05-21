//
// Created by Eric Irrgang on 2/26/18.
//

#ifndef HARMONICRESTRAINT_ENSEMBLEPOTENTIAL_H
#define HARMONICRESTRAINT_ENSEMBLEPOTENTIAL_H

#include <vector>
#include <array>
#include <mutex>

#include "gmxapi/gromacsfwd.h"
#include "gmxapi/session.h"
#include "gmxapi/context/outputstream.h"
#include "gmxapi/md/mdmodule.h"

#include "gromacs/restraint/restraintpotential.h"
#include "gromacs/utility/real.h"

#include "make_unique.h"

namespace plugin
{

// Histogram for a single restrained pair.
using PairHist = std::vector<double>;


// Stop-gap for cross-language data exchange pending SharedData implementation and inclusion of Eigen.
// Adapted from pybind docs.
template<class T>
class Matrix {
    public:
        Matrix(size_t rows, size_t cols) :
            rows_(rows),
            cols_(cols),
            data_(rows_*cols_, 0)
        {
        }

        explicit Matrix(std::vector<T>&& captured_data) :
            rows_{1},
            cols_{captured_data.size()},
            data_{std::move(captured_data)}
        {
        }

        std::vector<T> *vector() { return &data_; }
        T* data() { return data_.data(); };
        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
    private:
        size_t rows_;
        size_t cols_;
        std::vector<T> data_;
};

// Defer implicit instantiation to ensemblepotential.cpp
extern template class Matrix<double>;

/*!
 * \brief An active handle to ensemble resources provided by the Context.
 *
 * The semantics of holding this handle aren't determined yet, but it should be held as briefly as possible since it
 * may involve locking global resources or preventing the simulation from advancing. Basically, though, it allows the
 * Context implementation flexibility in how or where it provides services.
 *
 * Resources may be incoming input data or functors to trigger output data events.
 *
 * \internal
 * It is not yet clear whether we want to assume that default behavior is for an operation to be called for each edge
 * on every iterative graph execution, leaving less frequent calls as an optimization, or to fully support occasional
 * data events issued by nodes during their execution.
 *
 * In this example, assume the plugin has specified that it provides a `.ostream.stop` port that provides asynchronous
 * boolean events. We can provide a template member function that will handle either execution mode.
 *
 * ResourceHandle::ostream() will return access to a gmxapi::context::OutputStream object, which will provide
 * set("stop", true), to give access to a function pointer from a member vector of function pointers.
 *
 * In the case that we are triggering asynchronous data events, the function will make the appropriate call. In the case
 * that we have output at regular intervals, the function will update internal state for the next time the edge is
 * evaluated.
 *
 * In an alternative implementation, we could maintain a data object that could be queried by subscribers, but a publish
 * and subscribe model seems much more useful, optimizeable, and robust. We can issue the calls to the subscribers and
 * then be done with it.
 *
 * If we need to optimize for reuse of memory locations, we can do one of two things: require that
 * the subscribing object not return until it has done what it needed with the data (including deep copy) or use
 * managed handles for the data, possibly with a custom allocator, that prevents rewriting while there are read handles
 * still open. One way that came up in conversation with Mark to allow some optimization is to allow the recipient of
 * the handle to make either an `open` that gets a potentially blocking read-lock or an `open` that requests ownership.
 * If no other consumers of the data request ownership, the ownership can be transferred without a copy. Otherwise, a
 * copy is made.
 */
class EnsembleResourceHandle
{
    public:
        /*!
         * \brief Ensemble reduce.
         *
         * For first draft, assume an all-to-all sum. Reduce the input into the stored Matrix.
         * // Template later... \tparam T
         * \param data
         */
//        void reduce(const Matrix<double>& input);

        /*!
         * \brief Ensemble reduce.
         * \param send Matrices to be summed across the ensemble using Context resources.
         * \param receive destination of reduced data instead of updating internal Matrix.
         */
        void reduce(const Matrix<double> &send,
                    Matrix<double> *receive) const;

        /*!
         * \brief Issue a stop condition event.
         *
         * Can be called on any or all ranks. Sets a condition that will cause the current simulation to shut down
         * after the current step.
         */
        void stop();

        // to be abstracted and hidden...
        const std::function<void(const Matrix<double>&, Matrix<double>*)>* reduce_;

        gmxapi::Session* session_;

        /*!
         * \brief Get the current output stream manager.
         *
         * The output stream manager provides methods with signatures like `template<class T> set(std::string, T)` so
         * that a call to ostream()->set("stop", true) will find a registered resource named "stop" that accepts Boolean
         * data and call it with `true`.
         */
        gmxapi::context::OutputStream* ostream();
    private:
        std::shared_ptr<gmxapi::context::OutputStream> ostream_;
};

/*!
 * \brief Reference to workflow-level resources managed by the Context.
 *
 * Provides a connection to the higher-level workflow management with which to access resources and operations. The
 * reference provides no resources directly and we may find that it should not extend the life of a Session or Context.
 * Resources are accessed through Handle objects returned by member functions.
 *
 * One way to reframe this is to provide an array of function pointers to trigger the data events for which the plugin
 * is registered to provide. I think we could use a set of map structures: container for each function call signature,
 * which in our case is determined by the data type.
 */
class EnsembleResources
{
    public:
        explicit EnsembleResources(std::function<void(const Matrix<double>&, Matrix<double>*)>&& reduce) :
            reduce_(reduce)
        {};

        /*!
         * \brief Grant the caller an active handle for the currently executing block of code.
         *
         * EnsembleResourceHandle objects provide the accessible features to the client of the EnsembleResources.
         * If the naming doesn't seem right, please propose alternatives. A handle should not be moved, copied, or
         * held for any length of time. In other words, use a stack variable, preferably tightly scoped.
         *
         * \return Handle to current resources.
         */
         EnsembleResourceHandle getHandle() const;

         /*!
          * \brief Acquires a pointer to a Session managing these resources.
          *
          * \param session non-owning pointer to Session.
          */
         void setSession(gmxapi::Session* session);

         /*!
          * \brief Sets the OutputStream manager for this set of resources.
          *
          * \param ostream ownership of an OutputStream manager
          */
         void setOutputStream(std::unique_ptr<gmxapi::context::OutputStream> ostream);

    private:
        //! bound function object to provide ensemble reduce facility.
        std::function<void(const Matrix<double>&, Matrix<double>*)> reduce_;

        // Raw pointer to the session in which these resources live.
        gmxapi::Session* session_;

        // Shareable OutputStream object
        std::shared_ptr<gmxapi::context::OutputStream> ostream_;
};

/*!
 * \brief Template for MDModules from restraints.
 *
 * \tparam R a class implementing the gmx::IRestraintPotential interface.
 */
template<class R>
class RestraintModule : public gmxapi::MDModule // consider names
{
    public:
        using param_t = typename R::input_param_type;

        RestraintModule(std::string name,
                        std::vector<unsigned long int> sites,
                        const typename R::input_param_type& params,
                        std::shared_ptr<EnsembleResources> resources) :
            sites_{std::move(sites)},
            params_{params},
            resources_{std::move(resources)},
            name_{std::move(name)}
        {

        };

        ~RestraintModule() override = default;

        // \todo make member function const
        const char *name() override
        {
                return name_.c_str();
        }

        std::shared_ptr<gmx::IRestraintPotential> getRestraint() override
        {
                auto restraint = std::make_shared<R>(sites_, params_, resources_);
                return restraint;
        }

    private:
        std::vector<unsigned long int> sites_;
        param_t params_;

        // Need to figure out if this is copyable or who owns it.
        std::shared_ptr<EnsembleResources> resources_;

        const std::string name_;
};


struct ensemble_input_param_type
{
    /// distance histogram parameters
    size_t nBins{0};
    double binWidth{0.};

    /// Flat-bottom potential boundaries.
    double minDist{0};
    double maxDist{0};

    /// Experimental reference distribution.
    PairHist experimental{};

    /// Number of samples to store during each window.
    unsigned int nSamples{0};
    double samplePeriod{0};

    /// Number of windows to use for smoothing histogram updates.
    unsigned int nWindows{0};

    /// Harmonic force coefficient
    double k{0};
    /// Smoothing factor: width of Gaussian interpolation for histogram
    double sigma{0};

};

// \todo We should be able to automate a lot of the parameter setting stuff
// by having the developer specify a map of parameter names and the corresponding type, but that could get tricky.
// The statically compiled fast parameter structure would be generated with a recursive variadic template
// the way a tuple is. ref https://eli.thegreenplace.net/2014/variadic-templates-in-c/

std::unique_ptr<ensemble_input_param_type>
makeEnsembleParams(size_t nbins,
                   double binWidth,
                   double minDist,
                   double maxDist,
                   const std::vector<double> &experimental,
                   unsigned int nSamples,
                   double samplePeriod,
                   unsigned int nWindows,
                   double k,
                   double sigma)
{
    using gmx::compat::make_unique;
    auto params = make_unique<ensemble_input_param_type>();
    params->nBins = nbins;
    params->binWidth = binWidth;
    params->minDist = minDist;
    params->maxDist = maxDist;
    params->experimental = experimental;
    params->nSamples = nSamples;
    params->samplePeriod = samplePeriod;
    params->nWindows = nWindows;
    params->k = k;
    params->sigma = sigma;

    return params;
};

/*!
 * \brief a residue-pair bias calculator for use in restrained-ensemble simulations.
 *
 * Applies a force between two sites according to the difference between an experimentally observed
 * site pair distance distribution and the distance distribution observed earlier in the simulation
 * trajectory. The sampled distribution is averaged from the previous `nwindows` histograms from all
 * ensemble members. Each window contains a histogram populated with `nsamples` distances recorded at
 * `sample_period` step intervals.
 *
 * \internal
 * During a the window_update_period steps of a window, the potential applied is a harmonic function of
 * the difference between the sampled and experimental histograms. At the beginning of the window, this
 * difference is found and a Gaussian blur is applied.
 */
class EnsembleHarmonic
{
    public:
        using input_param_type = ensemble_input_param_type;

//        EnsembleHarmonic();

        explicit EnsembleHarmonic(const input_param_type &params);

        EnsembleHarmonic(size_t nbins,
                         double binWidth,
                         double minDist,
                         double maxDist,
                         PairHist experimental,
                         unsigned int nSamples,
                         double samplePeriod,
                         unsigned int nWindows,
                         double k,
                         double sigma);

        // If dispatching this virtual function is not fast enough, the compiler may be able to better optimize a free
        // function that receives the current restraint as an argument.
        gmx::PotentialPointData calculate(gmx::Vector v,
                                          gmx::Vector v0,
                                          gmx_unused double t);

        // An update function to be called on the simulation master rank/thread periodically by the Restraint framework.
        void callback(gmx::Vector v,
                      gmx::Vector v0,
                      double t,
                      const EnsembleResources &resources);

    private:
        /// Width of bins (distance) in histogram
        size_t nBins_;
        double binWidth_;

        /// Flat-bottom potential boundaries.
        double minDist_;
        double maxDist_;
        /// Smoothed historic distribution for this restraint. An element of the array of restraints in this simulation.
        // Was `hij` in earlier code.
        PairHist histogram_;
        PairHist experimental_;

        /// Number of samples to store during each window.
        unsigned int nSamples_;
        unsigned int currentSample_;
        double samplePeriod_;
        double nextSampleTime_;
        /// Accumulated list of samples during a new window.
        std::vector<double> distanceSamples_;

        /// Number of windows to use for smoothing histogram updates.
        size_t nWindows_;
        size_t currentWindow_;
        double windowStartTime_;
        double nextWindowUpdateTime_;
        /// The history of nwindows histograms for this restraint.
        std::vector<std::unique_ptr<Matrix<double>>> windows_;

        /// Harmonic force coefficient
        double k_;
        /// Smoothing factor: width of Gaussian interpolation for histogram
        double sigma_;
};

/*!
 * \brief Use EnsembleHarmonic to implement a RestraintPotential
 *
 * This is boiler plate that will be templated and moved.
 */
class EnsembleRestraint : public ::gmx::IRestraintPotential, private EnsembleHarmonic
{
    public:
        using EnsembleHarmonic::input_param_type;

        EnsembleRestraint(const std::vector<unsigned long> &sites,
                          const input_param_type &params,
                          std::shared_ptr<EnsembleResources> resources
        ) :
                EnsembleHarmonic(params),
                sites_{sites},
                resources_{std::move(resources)}
        {}

        std::vector<unsigned long int> sites() const override
        {
                return sites_;
        }

        gmx::PotentialPointData evaluate(gmx::Vector r1,
                                         gmx::Vector r2,
                                         double t) override
        {
                return calculate(r1, r2, t);
        };


        // An update function to be called on the simulation master rank/thread periodically by the Restraint framework.
        void update(gmx::Vector v,
                    gmx::Vector v0,
                    double t) override
        {
            // Todo: use a callback period to mostly bypass this and avoid excessive mutex locking.
            callback(v,
                     v0,
                     t,
                     *resources_);
        };

        /*!
         * \brief Implement the binding protocol that allows access to Session resources.
         *
         * The client receives a non-owning pointer to the session and cannot extent the life of the session. In
         * the future we can use a more formal handle mechanism.
         *
         * \param session pointer to the current session
         */
        void bindSession(gmxapi::Session* session) override
        {
            resources_->setSession(session);
        }

        void setResources(std::unique_ptr<EnsembleResources>&& resources)
        {
            resources_ = std::move(resources);
        }

    private:
        std::vector<unsigned long int> sites_;
//        double callbackPeriod_;
//        double nextCallback_;
        std::shared_ptr<EnsembleResources> resources_;
};


// Just declare the template instantiation here for client code.
// We will explicitly instantiate a definition in the .cpp file where the input_param_type is defined.
extern template class RestraintModule<EnsembleRestraint>;

} // end namespace plugin

#endif //HARMONICRESTRAINT_ENSEMBLEPOTENTIAL_H
