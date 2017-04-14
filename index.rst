.. vim: ts=3:sts=3

:tocdepth: 1

.. note::

    This document captures the redesign work of the ``lsst::afw::math::Statistics`` functionality.

.. warning::

    This is an initial draft. A partially working proof of concept implementation,
    which sketches out the general design and has been benchmarked to be equivalent
    to the current Statistics performance on 4k x 4k noise images, is available at:

    https://github.com/lsst/afw/u/pschella/DM-9979

.. sectnum::

Third party packages
====================

`ndarray`_: ``ndarray`` (``ndarray``).

.. _ndarray: https://github.com/ndarray/ndarray

LSST packages used
==================

`afw`_: ``lsst.afw.math`` (``afwMath``).
`afw`_: ``lsst.afw.image`` (``afwImage``).

.. _afw: https://github.com/lsst/afw

Requirements
============

Critical requirements From LDM-151 (chapter 9.7)
------------------------------------------------

#. Compute various robust statistics for central tendency and distribution widths, measured on 2-d and 1-d arrays.
#. Needs to be able to make use of mask and uncertainty arrays.
#. Needs to work on 2-D Images and MaskedImages.
#. Needs to work on stacks of aligned pixels for coaddition.

Keep from current implementation
--------------------------------

#. The implementation will be in C++.
#. Equivalent performance (but some micro optimizations may be dropped).
#. For efficiency multiple statistics will be computed together in a single pass wherever possible.
#. Enable statistics to be calculated for ``afw::image::Image<T>``, ``afw::image::MaskedImage<T>`` and ``std::vector<T>`` objects.
#. Support all currently supported statistics. I.e.
    - number of sample points
    - sample mean
    - sample standard deviation
    - sample variance
    - sample median
    - sample inter-quartile range
    - sample N-sigma clipped mean
    - sample N-sigma clipped stdev
    - sample N-sigma clipped variance
    - sample minimum
    - sample maximum
    - find sum of pixels in the image
    - find mean value of square of pixel values
    - or-mask of all pixels used
    - errors of requested quantities
#. Support statistics of ``afw::image::Mask<T>``.
#. Allow specification with bitmask which pixels to include / exclude in statistic.

Additional functionality in new implementation
----------------------------------------------

#. Change to ``ndarray::Array`` (instead of ``afw::image::Image<T>`` for type to operate on).
#. Configuration using ``pex::config``.
#. Allow for adding new statistics to calculate (at compile time, not (necessarily) run time),
   and enable current statistics to be calculated differently.
#. Add ability to see which pixels were clipped in output (e.g. as bitmask or as list of indices).
#. Ensure design is compatible with ``afw::detection::SpanSet::applyFunctor``.

Out of scope
------------

#. Adding new statistics.

Interfaces
==========

All statistics requests are handled through the ``afw::math::computeStatistics`` function 
(which roughly mimics `scipy.stats.describe`).
In Python its signature is as follows:

.. code-block:: python

    computeStatistics(img,
                       msk = None,
                       wgt = None,
                       var = None,
                       errors = False,
                       npoint = False,
                       mean = False,
                       stdev = False,
                       variance = False,
                       median = False,
                       iqrange = False,
                       meanclip = False,
                       stdevclip = False,
                       varianceclip = False,
                       min = False,
                       max = False,
                       sum = False,
                       meansquare = False,
                       ormask = False,
                       sctrl = StatisticsControl())

Where ``img``, ``msk``, ``wgt`` and ``var`` are typically a 1 dimensional ``numpy.ndarray``, or any of the other supported overloads (see below).

In C++ the signature is:

.. code-block:: cpp

    template <typename ImageT, typename MaskT, typename WeightT, typename VarianceT>
    StatisticsResult computeStatistics(ImageT const &img,
                              MaskT const *msk,     // (nullptr if not used)
                              WeightT const *wgt,   // (nullptr if not used)
                              VarianceT const *var, // (nullptr if not used)
                              int const flags,
                              StatisticsControl const &sctrl = StatisticsControl());

Where ``img``, ``msk``, ``wgt`` and ``var`` are typically ``ndarray::Array<T, 1, 1>`` of the appropriate type ``T``, and where ``flags`` is a bitwise combination of:

.. code-block:: cpp

    enum Property {
        NOTHING = 0x0,         ///< We don't want anything
        ERRORS = 0x1,          ///< Include errors of requested quantities
        NPOINT = 0x2,          ///< number of sample points
        MEAN = 0x4,            ///< estimate sample mean
        STDEV = 0x8,           ///< estimate sample standard deviation
        VARIANCE = 0x10,       ///< estimate sample variance
        MEDIAN = 0x20,         ///< estimate sample median
        IQRANGE = 0x40,        ///< estimate sample inter-quartile range
        MEANCLIP = 0x80,       ///< estimate sample N-sigma clipped mean (N set in StatisticsControl, default=3)
        STDEVCLIP = 0x100,     ///< estimate sample N-sigma clipped stdev (N set in StatisticsControl, default=3)
        VARIANCECLIP = 0x200,  ///< estimate sample N-sigma clipped variance
                               ///<  (N set in StatisticsControl, default=3)
        MIN = 0x400,           ///< estimate sample minimum
        MAX = 0x800,           ///< estimate sample maximum
        SUM = 0x1000,          ///< find sum of pixels in the image
        MEANSQUARE = 0x2000,   ///< find mean value of square of pixel values
        ORMASK = 0x4000        ///< get the or-mask of all pixels used.
    };

This is more natural in C++ for multiple options. An alternative would be to absorb the requested properties into ``StatisticsControl``.

Overloads
---------

The following overloads are present (also available from Python with the same syntax as above).

* Handle ``image::Image`` arguments with an optional ``image::Mask``.

.. code-block:: cpp

    template <typename ImagePixelT, typename MaskPixelT, typename WeightPixelT>
    StatisticsResult computeStatistics(lsst::afw::image::Image<ImagePixelT> const &img,
                              lsst::afw::image::Mask<MaskPixelT> const *msk,
                              lsst::afw::image::Image<WeightPixelT> const *wgt,
                              int const flags,
                              StatisticsControl const &sctrl = StatisticsControl());

* Handle ``image::MaskedImage``.

.. code-block:: cpp

    template <typename ImagePixelT, typename WeightPixelT>
    StatisticsResult computeStatistics(image::MaskedImage<ImagePixelT> const &mimg,
                              lsst::afw::image::Image<WeightPixelT> const *wgt,
                              int const flags,
                              StatisticsControl const &sctrl = StatisticsControl());

* Handle ``image::Mask``.

.. code-block:: cpp

    template <typename MaskPixelT>
    StatisticsResult computeStatistics(image::Mask<MaskPixelT> const &msk,
                              int const flags,
                              statisticscontrol const &sctrl = statisticscontrol());

These overloads mostly just pass through the respective ndarrays obtained with ``.getArray()``.

Configuration
-------------

Statistics to compute are selected either with keyword arguments (Python) or a bitmask (C++).
Additional configuration settings are provided through an (optional) ``StatisticsControl`` object
using standard functionality provided by ``pex_config``.

.. code-block:: cpp

    class NewStatisticsControl {
    public:
        NewStatisticsControl()
                : numSigmaClip(3),
                  numIter(3),
                  andMask(0x0),
                  noGoodPixelsMask(0x0),
                  isNanSafe(true),
                  calcErrorFromInputVariance(true),
                  baseCaseSize(100),
                  maskPropagationThresholds(16) {}
    
        LSST_CONTROL_FIELD(numSigmaClip, double, "Number of standard deviations to clip at");
        LSST_CONTROL_FIELD(numIter, int, "Number of iterations");
        LSST_CONTROL_FIELD(andMask, typename image::MaskPixel, "and-Mask to specify which mask planes to ignore");
        LSST_CONTROL_FIELD(noGoodPixelsMask, typename image::MaskPixel, "mask to set if no values are acceptable");
        LSST_CONTROL_FIELD(isNanSafe, bool, "Check for NaNs & Infs before running (slower)");
        LSST_CONTROL_FIELD(calcErrorFromInputVariance, bool,
                           "Calculate errors from the input variances, if available");
        LSST_CONTROL_FIELD(baseCaseSize, int, "Size of base case in partial sum for numerical stability");
        LSST_CONTROL_FIELD(
                maskPropagationThresholds, std::vector<double>,
                "Thresholds for when to propagate mask bits, treated like a dict (unset bits are set to 1.0)");
    };

Results
-------

Results are returned in the form of a single ``StatisticsResult`` object.
    
.. code-block:: cpp

    class StatisticsResult {
    public:
        size_t getNpoint() const;
        std::pair<double, double> getMean() const;
        std::pair<double, double> getStdev() const;
        std::pair<double, double> getVariance() const;
        std::pair<double, double> getMedian() const;
        std::pair<double, double> getIqrange() const;
        std::pair<double, double> getMeanClip() const;
        std::pair<double, double> getStdevClip() const;
        std::pair<double, double> getVarianceClip() const;
        double getMin() const;
        double getMax() const;
        double getSum() const;
        typename image::MaskPixel getOrMask() const;
    };

.. note::

    Non-computed results are indicated with ``NaN``.

.. note::

    In Python all getters are also available as properties.

Implementation
==============

The above requirements lead to the following design guidelines:

    #. Use compile-time polymorphism (through templated types) exclusively (at least in the inner loop) for abstraction, configurability and speed.
    #. Translate runtime options to compile-time constants (such that branches in the inner loop are compiled away) as much as possible.
    #. Enable pairwise summation for numerical stability.
    #. Separate accumulation of intermediate products from calculating of final values (to enable interoperability with ``afw::detection::SpanSet::applyFunctor``).

Design overview
---------------

The general design uses a Strategy like structure but with compile-time (instead of run-time) polymorphism.
It has the following components:

    - ``Algorithm``: calculates (``collect``) and combines (``combineWith``) intermediate values and produces the final ``StatisticsResult`` (``reduce``).
      Relies on external state (``ExternalData``) for nested runs.
      Only ``StandardStatistics`` is implemented, but can be swapped out later.

    - ``Runner<Algorithm, Validator>``: performs (``operator()``) a single pass run of the algorithm on the data.
      Holds a (unique) instance of ``ExternalData``.
      Only ``SinglePassStatistics`` is implemented, but can be swapped out later.

    - ``Validator``: checks if pixels should be included or excluded.
      Are composable with various simple building blocks available.

Internal API
------------

computeStatistics
^^^^^^^^^^^^^^^^^^

The externally visible ``computeStatistics`` function calls an internal ``detail::translateOptions``,
which has a series of overloads that reduce runtime function arguments to compile-time template parameters.
At the end of this series this results in a call to ``detail::computeStatistics``.

.. code-block:: cpp

    namespace detail {

    template <bool useMask, bool useWeight, bool useVariance, bool computeRange, bool computeMedian,
              bool sigmaClipped, typename ImageT, typename MaskT, typename WeightT, typename VarianceT>
    StatisticsResult computeStatistics(ImageT const &img, MaskT const &msk, WeightT const &wgt,
                              VarianceT const &var, NewStatisticsControl const &sctrl);

    }  // namespace detail

At this point one run (or multiple runs in case of sigma clipping) of ``SinglePassStatistics`` are performed
and the results are returned.

.. note::

    Currently the code assumes ``StandardStatistics`` as the underlying algorithm, but this can be
    changed to a template parameter if needed.

SinglePassStatistics
^^^^^^^^^^^^^^^^^^^^

The default runner is ``SinglePassStatistics``. This implementation uses pairwise summation (combination)
down to a ``baseCaseSize`` after which it performs a simple linear run through the data.
Internally it keeps one instance of ``Algorithm::ExternalData`` which is passed to the ``Algorithm::collect``
and ``Algorithm::reduce`` functions (by reference) where needed.

It takes the following template parameters:

    - ``Validator``: A functor that takes image, mask, weight and variance arguments and returns ``true``
      if the value is to be clipped, or ``false`` otherwise.
    - ``Algorithm``: The statistics algorithm to execute.

Constructors:

    - ``explicit SinglePassStatistics(const Validator &validator = Validator(), size_t baseCaseSize = 100)``
      Copies of both input parameters are held as private members.
    - ``default`` copy and move constructors (as well as copy and move assignment operators).

Member functions and operators:

    - ``operator()``: Run ``Algorithm`` once on provided ``img``, ``msk``, ``wgt`` and ``var``
      using pairwise summation / combination.

    .. code-block:: cpp

        template <typename ImageT, typename MaskT, typename WeightT, typename VarianceT,
                  typename... AlgorithmArgs>
        StatisticsResult operator()(ImageT const &img, MaskT const &msk, WeightT const &wgt,
                          VarianceT const &var);

StandardStatistics
^^^^^^^^^^^^^^^^^^

The default algorithm is represented by the ``StandardStatistics`` class which has the following API

Member type:

    - ``ExternalData`` : Algorithm specific data that has to be stored across different instances of ``StandardStatistics``,
      to be provided by the runner.

Template Parameters:

    - ``useMask``, ``bool`` : Use the provided mask (assume ``msk`` is ``UnusedParameter`` if not)
    - ``useWeight``, ``bool`` : Use the provided weights (assume ``wgt`` is ``UnusedParameter`` if not) 
    - ``useVariance``, ``bool`` : Use the provided variance (assume ``var`` is ``UnusedParameter`` if not) 
    - ``computeRange``, ``bool`` : Compute ``min`` and ``max`` values (can be split if desired)
    - ``computeMedian``, ``bool`` : Compute the ``median`` (requires ``ExternalData`` to store a copy of good pixel values)

Constructors:

    - Default constructor, ``default`` copy and move constructors (as well as copy and move assignment operators).

Member functions:

    - ``collect``: Gather intermediate products, can be run multiple times to accumulate one or more values.

        .. code-block:: cpp

            template <typename ImageIter, typename MaskIter, typename WeightIter, typename VarianceIter,
                      typename Validator>
            void collect(ImageIter img, MaskIter msk, WeightIter wgt, VarianceIter var, size_t n,
                         Validator const &validator, ExternalData &externalData);
    
    - ``reduce``: Compute intermediate results (e.g. ``mean``, ``median``, or ``variance``, but not ``stdev`` which is computed lazily from the variance)

        .. code-block:: cpp
        
            StatisticsResult reduce(ExternalData &externalData);

    - ``combineWith``: Combine intermediate products.

        .. code-block:: cpp
        
            void combineWith(const StandardStatistics &rhs);

Validators
^^^^^^^^^^

During a run of ``Algorithm::collect`` each set of ``img``, ``msk``, ``wgt`` and ``var`` values
is passed to the provided ``Validator`` functor. If its return value is ``false`` the pixel is clipped.

Validators are composable. The ``makeCombinedChecker`` variadic function creates a single validator out of its provided arguments.

It does this by nesting instances of ``CheckBoth`` which itself takes two template arguments ``First`` and ``Second`` and has an ``operator()`` that returns the logical and of ``First::operator()`` and ``Second::operator()``.

The following building blocks are provided:

    - ``AlwaysTrue``: always returns ``true``;
    - ``AlwaysFalse``: always returns ``false``;
    - ``CheckMask``: returns bitwise and of ``msk`` value with ``mask`` argument provided at construction;
    - ``CheckFinite``: returns ``true`` if ``img`` is finite (e.g. not ``NaN`` or infinity);
    - ``CheckRange``: returns ``true`` if ``img`` is within the interval ``center +/- limit`` where
      ``center`` and ``limit`` are provided at construction.

Unused parameters
^^^^^^^^^^^^^^^^^

Unused parameters (for ``msk``, ``wgt`` or ``var``) are given as ``nullptr`` (C++) or ``None`` (Python) in the API, but for lower layers are given as instances of ``UnusedParameter``.
A fake vector / iterator that does nothing.
This makes the inner loop in ``Algorithm::collect`` straightforward while compiling away into non-existence..

