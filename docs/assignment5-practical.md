# Practical Report on the Jenkins CI Pipeline

## Project and objective

The practical task uses the dissertation repository “AcademicRecomendationSystem” rather than a sample Iris classifier. It is a Python/FastAPI platform that enriches researcher profiles from scholarly sources and ranks publications, venues, and potential collaborators. The repository is available at https://github.com/AslanAbishev/AcademicRecomendationSystem. The objective is to automatically run software checks and build a Docker image whenever a new commit is detected by Jenkins.

## Implemented configuration

The existing `Dockerfile` builds the application image. The added `Jenkinsfile` defines checkout, a lightweight Docker build (`INSTALL_ML=false`), Python syntax compilation, pytest, and a `/health` smoke test. The lightweight build is deliberate: SciBERT, SPECTER2, and GraphSAGE dependencies and model weights are too large for a fast per-commit check. The added `Dockerfile.jenkins` and `docker-compose.jenkins.yml` start a local Jenkins server with the Docker CLI. `docs/assignment5-jenkins-guide.md` gives setup, evidence, and shutdown instructions. The project also has an existing GitHub Actions workflow, but this report treats Jenkins as the Assignment 5 implementation.

## Execution and evidence status

The files are prepared in the local working tree. A successful Jenkins run cannot yet be claimed: Docker Desktop’s Linux engine was unavailable during preparation, and the new files have not been committed or pushed. Locally, 54 tests covering data quality, recommendation logic, and registration passed, while the full suite stalled during API tests in this environment; the Jenkins run is needed to establish whether this is an environment-specific issue or a test defect. Therefore, a screenshot of a successful pipeline and a GitHub link to the new Jenkinsfile must be added only after Docker is running, Jenkins has completed a build, and the files are pushed. The checklist is: start Docker Desktop; run `docker compose -f docker-compose.jenkins.yml up --build -d`; complete the setup wizard at `http://localhost:8080`; create a Pipeline job from the Git repository with script path `Jenkinsfile`; run Build Now; inspect the stage view and console output; commit and push the new files; capture the actual green result and repository page. Jenkins polls SCM every five minutes while running; local Jenkins cannot receive public GitHub webhooks without a reachable endpoint.

## Difficulties and conclusions

The main operational difficulty is the dependency on a working Docker daemon. The pipeline itself should be validated by a real Jenkins execution before submission. Another research-specific difficulty is avoiding a false equivalence between passing software tests and proving recommendation quality. This pipeline checks code and container behavior; it does not retrain models or evaluate NDCG, coverage, or fairness on a frozen temporal dataset. Those are separate experiments. Mounting the Docker socket into Jenkins and running the controller as root is acceptable only for a local coursework demonstration and should not be used for a public deployment. Overall, the configuration creates a reviewable CI process, but completion evidence remains conditional on an actual successful run.

## Screenshots to insert after verification

1. Jenkins job page showing the successful pipeline stages.
2. Console output showing the pytest summary and `Finished: SUCCESS`.
3. GitHub repository page showing the committed `Jenkinsfile` and its URL.
