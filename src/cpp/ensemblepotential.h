//
// Created by Eric Irrgang on 2/26/18.
//

#ifndef HARMONICRESTRAINT_ENSEMBLEPOTENTIAL_H
#define HARMONICRESTRAINT_ENSEMBLEPOTENTIAL_H

#include <array>
#include <functional>
#include <mutex>
#include <type_traits>
#include <vector>

#include "gmxapi/gromacsfwd.h"
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
         * \brief Apply a function to each input and accumulate the output.
         *
         * \tparam I Iterable.
         * \tparam T Output type.
         * \param iterable iterable object to produce inputs to function
         * \param output structure that should be present and up-to-date on all ranks.
         * \param function map each input in iterable through this function to accumulate output.
         */
        template<typename I, typename T>
        void map_reduce(const I& iterable, T* output, void (*function)(double, const PairHist&, PairHist*));

        // to be abstracted and hidden...
        const std::function<void(const Matrix<double>&, Matrix<double>*)>* _reduce;
};

/*!
 * \brief Reference to workflow-level resources managed by the Context.
 *
 * Provides a connection to the higher-level workflow management with which to access resources and operations. The
 * reference provides no resources directly and we may find that it should not extend the life of a Session or Context.
 * Resources are accessed through Handle objects returned by member functions.
 */
class EnsembleResources
{
    public:
        EnsembleResourceHandle getHandle() const;

        class Builder;

    private:
        EnsembleResources(std::function<void(const Matrix<double>&, Matrix<double>*)>&& reduce,
                          double callBackPeriod,
                          bool hasCallBack) :
            reduce_{std::forward<decltype(reduce_)>(reduce)},
            callBackPeriod_{callBackPeriod},
            hasCallBack_{hasCallBack}
        {};

        //        std::shared_ptr<Matrix> _matrix;
        std::function<void(const Matrix<double>&, Matrix<double>*)> reduce_;
        double callBackPeriod_;
        bool hasCallBack_{false};
};

class EnsembleResources::Builder
{
    public:
        std::shared_ptr<EnsembleResources> build()
        {
            //auto resources = std::make_shared<EnsembleResources>(std::move(functor_), callBackPeriod_, hasCallBack_);
            EnsembleResources new_resources{std::move(functor_), callBackPeriod_, hasCallBack_};
            auto resources = std::make_shared<EnsembleResources>(std::move(new_resources));
            return resources;
        };

        Builder& setFunctor(std::function<void(const Matrix<double>&, Matrix<double>*)>&& functor)
        {
            functor_ = std::move(functor);
            return *this;
        }

        Builder& setCallBackPeriod(double t)
        {
            callBackPeriod_ = t;
            hasCallBack_ = true;
            return *this;
        }

    private:
        std::function<void(const Matrix<double>&, Matrix<double>*)> functor_;
        double callBackPeriod_;
        bool hasCallBack_{false};
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

template<class T>
struct has_callback
{
    private:
        typedef std::true_type yes;
        typedef std::false_type no;

        template<typename U> static auto test() -> decltype(U::callback(), yes());
        template<typename> static no test();

    public:
        static constexpr bool value = std::is_same<decltype(test<T>()), yes>::value;
};

/*!
 * \brief Use EnsembleHarmonic to implement a RestraintPotential
 *
 * This is boiler plate that will be templated and moved.
 */
template<class PotentialT>
class Restraint : public ::gmx::IRestraintPotential, private PotentialT
{
    public:
        using input_param_type = typename PotentialT::input_param_type;
        using PotentialT::calculate;

        using callbackType = std::function<void(gmx::Vector, gmx::Vector, double, const EnsembleResources &)>;

        Restraint(std::vector<unsigned long> sites,
                  const input_param_type& params,
                  std::shared_ptr<EnsembleResources> resources
        ) :
                PotentialT(params),
                sites_{std::move(sites)},
                callbackPeriod_{},
                nextCallback_{},
                resources_{std::move(resources)}
        {}

        std::vector<unsigned long int> sites() const override
        {
                return sites_;
        }

        // todo: template on call signature availability using decltype or maybe an SFINAE check in the return type
        gmx::PotentialPointData evaluate(gmx::Vector r1,
                                         gmx::Vector r2,
                                         double t) override
        {
                return calculate(r1, r2, t);
        };


        // An update function to be called on the simulation master rank/thread periodically by the Restraint framework.
        // The function is redefined with a specialization below when PotentialT implements a calculate() member function.
        void update(gmx::Vector v,
                    gmx::Vector v0,
                    double t) override;

        double nextUpdateTime() const override
        {
            return nextCallback_;
        }

        void setResources(std::unique_ptr<EnsembleResources>&& resources)
        {
            resources_ = std::move(resources);
        }

    private:

        std::vector<unsigned long int> sites_;
        double callbackPeriod_;
        double nextCallback_;
        std::shared_ptr<EnsembleResources> resources_;

        template<typename T>
        auto dispatchCallback(gmx::Vector v,
                              gmx::Vector v0,
                              double t) -> decltype(T::callback(v, v0, t, *resources_))
        {
            T::callback(v, v0, t, *resources_);
        }

        // If the PotentialT defines both callback signatures, both of these template signatures
        // will match and bad things could happen.
        template<typename T>
        auto dispatchCallback(gmx::Vector v,
                              gmx::Vector v0,
                              double t) -> decltype(T::callback(v, v0, t))
        {
            T::callback(v, v0, t);
        }
};

template<class PotentialT>
void Restraint<PotentialT>::update(gmx::Vector v,
            gmx::Vector v0,
            double t)
{
    if (t >= nextCallback_)
    {
        dispatchCallback<PotentialT>(v, v0, t);
        nextCallback_ += callbackPeriod_;
    }
};

// Just declare the template instantiation here for client code.
// We will explicitly instantiate a definition in the .cpp file where the input_param_type is defined.
extern template class RestraintModule<Restraint<EnsembleHarmonic>>;

} // end namespace plugin

#endif //HARMONICRESTRAINT_ENSEMBLEPOTENTIAL_H
