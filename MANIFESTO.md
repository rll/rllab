# rllab2 Manifesto
Clarity of purpose is the single largest impediment to effective software development. The purpose of this document is to establish clarity on philosophical and practical matters related to the development of the next version of rllab.

This plan might be implemented by modifying rllab under its existing namespace and repository, by adding a new namespace to the existing repository, or by forking rllab entirely into a new namespace and repository (see [Important Decisions](#important-decisions)).

## Purpose
rllab2 is a software toolkit for implementing, characterizing, comparing, and communicating reinforcement learning algorthms, with a particular emphasis on applications to robotics.

It is designed to accelerate the development and dissemination of RL algorithms by providing a toolkit of high-quality, interoperable, reusable components which are used to implement RL algorithms. Accompanying the toolkit is a library of high-quality reference implementations of RL algorithms which use rllab2.

rllab2 is specifically designed to promote reproducible research, by providing a stable library upon which researchers can build and share their work. In an ideal world, any RL researcher should be able to phrase the code implementing her paper as a pull request against rllab2.

### Who is the user?
The target users of this software are _reinforcement learning researchers_. Our users are also our primary developers, and the software is designed first-and-foremost to serve the needs of its developers.

Other users are welcome and encouraged to use and improve the software, but the developers will not offer technical support for debugging basic system configuration and installation issues which do not constitute flaws in the codebase. Likewise, the developers, issue tracker, and wiki are not a resource for help learning the basics of reinforcement learning.

rllab2 is not designed for use in production environments, which necessarily prioritize security, performance, stability, and explicitness. Contributions which make rllab2 more suitible for use in production environments are always welcome, but design decisions will prioritize research velocity.

## Philosophy
* __Prioritize ergnomics and clarity__
  rllab2's purpose is to accelerate research. The best way to communicate new algorithms is with code, so algorithms implemented with rllab2 should be easy to read. Research is usually time-constrained, and testing an idea quickly is usually more important than eliminating all ambiguity in an implementation, so rllab2 should provide sane defaults wherever possible. In other words, the ideal number of required arguments to a function or constructor is zero. rllab2's primitives should be reusable and comprehensive-enough that implementing a new algorithm from a paper is a matter of writing it down using rllab2 primitives (and perhaps adding a new primitive or two, or enhancing an existing primitive). This will not only accelerate replication of work, it will also encourage new work be contributed back to the library.  

* __Gradual abstraction and the linear learning curve__
  Simple things should be easy, and hard things should be possible. The amount of code necessary to achieve something should be approximately proportional to its complexity and specificity. Sometimes the details truly do matter. rllab2 should provide interfaces for specifying the minute details of an implementation, but they should be hidden from users which have no preference about the particulars of a component's behavior.

* __Pay only for what you use__
  Every research project is different, and will use a different subset of rllab2. The library should only consume system resources or user cognition for components which are actually used. For instance, a worker process for plotting should not be launched unless the plotter is actually instantiated. Features and configuration which enhance performance or specificity at the expense of complexity and expressiveness are _always optional_.

* __Plays well with others__
  An RL software system has lots of moving parts, and RL research usually involves modifying one or more of those parts to test new ideas. rllab2 components should be able to operate in isolation from one another whenever possible, so researchers are not presented with an "all or nothing" choice between rllab2 lock-in and a homegrown fork. Most research projects will involve modifying or replacing some part of the library. rllab2 primitives should be good starting points for extension and modification, and not over-optimized for existing use cases. Contributing modified, derived, and new APIs should be easy.

* __Better together__
  Though useful in isolation, rllab2 components should work seamlessly together whenever possible.

* __You (probably) don't know better__
  Where the community has already converged on a standard interface or tool for a particular purpose, use it. Nobody benefits from the proliferation of multiple irrelevantly-different standards for simple ideas. If the community concensus choice is insufficient for your purposes, first try to amend it and contribute it back. Then try to wrap or extend it. Only fork and modify it as a last resort. Do not hide perfectly-sufficient abstractions from the community behind your own shims just to achieve what you consider a more aesthetically-pleasing result. This leads rapidly to the [inner-platform effect](https://en.wikipedia.org/wiki/Inner-platform_effect)

* __Quality is a priority__
  The quality of software is important. The quality of a library is essential to its usefulness--users need to be able to make the assumption that importing the library is much more likely to yield a correrct implementation than trying to write it themselves. rllab2's code should be readable, well-documented, and conform to a well-defined stylistic standard. Testing software of this sort is challenging, but it should be employed wherever practical. Fixing the small bugs and warnings is important too. When a user has to constantly work around easy-to-fix and loud flaws in a software library, it's reasonable for them to suspect that other parts are silently failing in ways that will sabotage their research.


## Implementing the Philosophy

### Explicit Trade-Offs
Use these as a guide for making difficult design decisions which have real trade-offs, and to avoid bike-shedding and flame wars. This list should be amended as the new trade-offs are identified and concensus formed about which side rllab2 lands on.

* __Simplicity, ergonomics, and clarity over performance, security, explicitness, backward-compatiblity, and pretty much any other production software concern.__

* __Minimalism__
  Sometimes it is tempting to implement a fundamental function of the library (e.g. multiprocess training) by shoehorning our use case into the API of some very large third-party dependency which was never really intended for that purpose (e.g. Distributed TensorFlow). In these cases, we prefer to find a much smaller library for our use case, or write our own implementation of that functionality instead. This choice extends recursively to the third-party dependencies we choose to adopt: we prefer minimalist libraries.

### Important Decisions
This section documents important, specific technical decisions with a large impact on the scope and implementation of rllab2. It should be amended as decisions are made and new important decisions are identified.

#### Made
* __`gym.env` is the environment abstraction of choice for rllab2.__
  rllab2 primitives with an environment dependency will accept `gym.Env` objects for that argument. `rllab.envs.Environment` should be gradually deprecated and eventually removed. Note that this is only a dependency on the `gym.Env` interface, not the associated environments, registration system, physics engines, benchmarking suite, etc. Existing wrappers to e.g. `box2d`, raw MuJoCo, and `dm_control` will be maintained, but they will wrap to a `gym.Env` interface.

* __We will remove the `lasagne`/`layers` library and associated home-grown neural network primitves from rllab2.__
  Most primitives depending on a neural network should be implemented as fairly-flat blobs of code in the native neural network library of rllab2. We can also maintain a flat utility library with common composite operations and subgraphs, which uses the native types of the underlying neural network library.

* __Python 3 is the only supported Python for rllab2.__

#### Pending
* __Fork vs amend__
  Implementing rllab2 means deleting, rewriting, and/or reorganizing large parts of the original rllab codebase. This will require the cooperation and support of the original rllab maintainers. Rewriting the software will create some disorder for current users and require more attention to backwards-compatibility on the part of the developers, but it maintains continuity behind a brand, retains a sizable GitHub following, and sets a good example for productive open source software collaboration. Forking rllab into a new repository avoids many of these complications, but is disruptive to the prospective user base.

* __conda vs pip__
  We need to decide how rllab2 will appear as a library. The current library uses conda to create a well-isolated "workbench" style environment, at the expense of making it difficult to integrate with non-rllab software. Recasting rllab as a standard Python package would make it easier use rllab as a standalone library, but makes dependency management a little trickier, especially for heavy, core dependencies.

* __TensorFlow vs PyTorch__
  We need to choose a neural network library and stick to it. This is a deep platform dependency and is difficult to extricate later, so the logical choices are those which are supported by a very well-resourced third-party, which is unlikely to abandon it any time soon. It should also enjoy widespread use in the RL research community. That narrows the field to TensorFlow and PyTorch.

## Making Contributions

### Development Workflow
* rllab2 will use GitHub as its authoritative repository and tracker.
* We will maintain a linear commit history with rebase-only merging (i.e. no merge commits), tutorial [here](https://gist.github.com/markreid/12e7c2203916b93d23c27a263f6091a0).
* All pull requests must be based on the `master` branch, and rebase cleanly on that branch. 
* No pull request will be merged without (1) passing automated quality checks from the CI and (2) addressing the feedback of and receiving approval from a maintainer (in that order). 

### Style, Code Quality, and Testing
* rllab2 will use [PEP8](https://www.python.org/dev/peps/pep-0008/) style, defined as the output of the most recent release of [yapf](https://github.com/google/yapf) in PEP8 mode. This includes the [PEP8 rules on import ordering](https://www.python.org/dev/peps/pep-0008/#imports). Pull requests not conforming to the style check will not be merged. Pull requests which modify an existing non-conforming file should bring that file up-to-spec.
* Readability is important. Algorithm and API implementations should be implemented using terminology and organization as closely as possible to their source papers.
* We may add additional automated quality checks using automatic tools such as pylint, configured to enforce rules which are important for our use case.
* We will write tests for the software and run them automatically where we can. Testing is easiest (and most important) for primitive modules, and gradually more challenging for larger composites of those modules.
* New algorithms with an existing canonical implementation (e.g. in [openai/baselines](https://github.com/openai/baselines)) should be accompanied by an automated regression test against that implementation. New algorithms without an existing canonical implementation should be accompanied by an automated regression test with appropriate admittance criteria (e.g. average performance on a benchmark test, convergence time, etc.).
* There will be no submodules in the rllab2 git repository.

## Details
### Platform Support
rllab2's target platform is the two most recent LTS releases of Ubuntu Linux, which are 16.04 and 18.04 at the time of writing.

Support for non-Ubuntu Unixen, especially Mac OS X, is desirable and will be maintained on a best-effort basis by fixing platform-specific issues identified by the users of those platforms. Issue reports for non-Ubuntu platforms which are not accompanied by a pull request or specific fix instructions will be ignored.

We will not prioritize support for Windows and other niche platforms, but we welcome pull requests which improve support for other platforms, and contributors who wish to maintain that support. We will intentionally remove support for platforms which no longer have a maintainer, since no support is strictly better than unreliable support.

A GPU is not required to run rllab2.

### Releases and Backwards Compatibility
rllab2 will break backward compatibility with rllab where necessary to make improvements, but not arbitrarily.

We will attempt to follow a thrice-yearly major release schedule, aligned to the predominant North American academic schedule and numbered by month and year, i.e. the next releases are 18.05, 18.09, and 19.01. Updates within a release version will be backwards compatible if at all possible. Changes between releases will prioritize improvement over backwards-compatibility.

### Attribution
All components and algorithms in rllab2 which make use of open source or scholarly work will provide explicit in-code and in-documentation attribution to the source of that work.

### Dependencies
Every dependency to the package must justify its existence. The optimal number of dependencies for any software project is 0, but we will have to settle for the smallest number we can afford.

A new dependency adds complexity, overhead, and a new point of failure outside of the project's control. Large dependencies should not be pulled into rllab2 just to provide a few helper functions which could otherwise be duplicated.

Do not engage in complex reductions of our use case just to make use of a third-party dependency, especially if that dependency is large. Just write your own stripped-down version.

All third-party dependencies will be precisely versioned and kept up-to-date with best effort.

Choices of major dependencies will prioritize libraries which are known to be stable, well-documented, well-maintained (and likely to stay that way), and as small as possible.