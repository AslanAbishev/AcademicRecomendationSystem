# The Role of DevOps and CI/CD in Automating a Research Project on Academic Profile Promotion

## Introduction

This report examines how DevOps practices support the dissertation project “Machine Learning Model for Personalized Promotion of an Early-Career Researcher’s Scientific Profile in Digital Academic Networks.” The project combines a FastAPI application, OpenAlex-based scholarly data, PostgreSQL with pgvector, several text and graph ranking methods, and a web interface. Its research results depend not only on model quality, but also on whether another person can reproduce data preparation, tests, and software deployment. DevOps is relevant here because a broken data pipeline or an untested code change can invalidate a comparison between recommendation models even when the mathematical idea is sound.

## 1. DevOps goals and principles

DevOps is a set of practices that reduces the separation between software development and operation. Its practical principles are version control, automated checks, reproducible environments, short feedback loops, monitoring, and shared responsibility for failures. In a research project, these principles have an additional scientific purpose: the program version, dependencies, data snapshot, model artifact, and evaluation protocol should be traceable together. A claim such as “model A outperforms model B” is hard to verify if the evaluation script or candidate pool changed between runs.

For this dissertation, Git records the source code and reviewable changes; Docker defines a repeatable runtime; tests catch regressions in API and recommendation logic; and CI repeats those checks after commits. Research data and large model files are not stored in Git, so reproducibility additionally requires dataset manifests, source identifiers, acquisition dates, temporal cutoffs, and model checksums. Containerization alone does not make an experiment scientifically reproducible: an external API can return changed records, model weights can differ, and some computations can be nondeterministic.

## 2. Continuous Integration and Continuous Delivery

Continuous Integration (CI) merges changes frequently and automatically checks each revision. A typical pipeline checks out the repository, installs dependencies, validates syntax or style, runs unit tests, and builds an image. Continuous Delivery (CD) goes further by preparing a tested artifact for release; continuous deployment additionally releases it without a manual gate. The distinction matters: a successful CI run says the tested behavior works under specified conditions, not that the model is effective for real researchers or that the service is safe to deploy publicly.

The project already has a GitHub Actions workflow that runs Python checks, pytest, and a lightweight Docker build. Assignment 5 separately asks for Jenkins unless an alternative is approved. The added Jenkinsfile gives the same repository an explicit, versioned Jenkins CI path: checkout, Docker build with heavyweight optional ML dependencies disabled, syntax checks, unit tests, and an API health smoke test. This lean image avoids downloading transformer weights for every commit; model-training and scientific benchmark jobs should remain separate, slower workflows tied to frozen data and explicit resource budgets.

For a public research platform, CD should require more than a green unit-test pipeline. A defensible gate would include dependency and secret scanning, schema migration checks, an integration test with PostgreSQL/pgvector, performance and fairness regression thresholds, and manual approval before production release. Model metrics such as NDCG@10, coverage, and exposure gap must be reported alongside dataset version and evaluation cutoff. The current Jenkins pipeline does not yet implement these release gates.

## 3. DevOps tools in a research environment

Docker packages the API and its Python dependencies into an image defined by a Dockerfile. Docker Compose expresses the local multi-service environment: API, PostgreSQL/pgvector, Redis, and an optional ML service. This helps a researcher start the same application stack on different machines and keeps service configuration visible. It does not freeze the OpenAlex API response or private user data; data provenance remains a separate concern.

Jenkins is an automation server whose Pipeline-as-Code approach stores CI instructions in a Jenkinsfile next to the research code. Stages make failures visible: a broken checkout differs from a failing test or image build. In the local coursework setup, Jenkins runs in a container and uses the host Docker socket to build and test the API image. This is convenient for demonstration but grants broad host control, so it must not be exposed to the Internet or mistaken for a hardened production architecture. A local Jenkins instance can poll the repository for commits; GitHub cannot send a webhook to localhost unless a reachable endpoint is provided.

Kubernetes orchestrates containers across machines and can manage replicas, restarts, service discovery, and rolling updates. For this dissertation it is a possible future deployment platform, not a required component of the current local research setup. Adopting it before the model, data, and user-study protocol are stable would add operational complexity without improving experimental validity. Its value becomes clearer if real participants require reliable access, isolation of batch embedding jobs from the web service, and controlled rollout of model versions.

Version-control and CI services also improve collaboration. A commit fixes the exact source revision used in an experiment, pull-request checks expose problems before merge, and pipeline logs document which checks ran. However, external datasets, database state, secrets, and pretrained-model licenses need their own governance. The OpenAlex API key belongs in an environment variable or secret store, never in the repository or build log. User accounts and academic identifiers require access controls and a privacy policy before a public pilot.

## 4. Concrete applications to the dissertation system

First, a developer modifies collaborator ranking. CI runs tests that exercise API contracts and recommendation logic, then builds the same lightweight image used for demonstration. A failing test prevents the change from being treated as verified. Second, a researcher retrains a graph-text model on a frozen temporal snapshot. The training job should record input-file hashes, preprocessing version, random seed, model configuration, and validation results. It must not optimize on a previously inspected final holdout and then report that holdout as independent. Third, a pilot participant opens the platform, connects identifiers, and receives recommendations. Application monitoring should distinguish a recommendation displayed, opened, acted upon, and later externally verified; a click or self-reported completion is not proof of increased scientific visibility.

These examples show that DevOps supports both software quality and scientific integrity. The currently available offline evaluation can compare ranking methods, but real promotion impact requires prospective observation, consent, and clearly defined outcome measures. CI tests protect implementation consistency; they do not replace research design.

## Conclusion

DevOps addresses recurring research-software problems: environment drift, unnoticed regressions, undocumented changes, and fragile handoffs. In this project, Docker, Git, automated tests, GitHub Actions, and the added Jenkins pipeline provide a foundation for repeatable software delivery. The next maturity step is to connect CI to versioned research artifacts and scientifically justified quality gates, while keeping model validation separate from software smoke tests. Kubernetes and automatic CD are optional later steps, not evidence of novelty by themselves. The dissertation’s contribution must be judged by a clearly specified recommendation method and a valid evaluation, not by the number of deployment tools used.

## References

1. Jenkins. Pipeline documentation. https://www.jenkins.io/doc/book/pipeline/
2. Jenkins. Declarative Pipeline syntax. https://www.jenkins.io/doc/book/pipeline/syntax/
3. Docker. Building container images. https://docs.docker.com/guides/lab-building-images/
4. Kubernetes. Concepts. https://kubernetes.io/docs/concepts/
5. Project repository. https://github.com/AslanAbishev/AcademicRecomendationSystem
