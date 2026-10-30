# Assignment 5: Jenkins CI demonstration

This coursework setup runs Jenkins locally with access to the Docker Desktop Linux engine. It is for local demonstration, not production deployment. Mounting the Docker socket and running Jenkins as root gives the container host-level control; never expose this instance to the public Internet.

## Prerequisites

- Docker Desktop is running in Linux-container mode.
- The repository is available at `https://github.com/AslanAbishev/AcademicRecomendationSystem`.
- Ports 8080 and 50000 are free. The application itself uses port 8000.

## Start Jenkins

From the repository root:

```powershell
docker compose -f docker-compose.jenkins.yml up --build -d
docker compose -f docker-compose.jenkins.yml logs -f jenkins
docker compose -f docker-compose.jenkins.yml exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

Open `http://localhost:8080`, paste the password, and complete the setup wizard. Create a `Pipeline` job, select **Pipeline script from SCM**, choose **Git**, enter the repository URL, branch `*/assignment-3-ci` (the current project branch), and script path `Jenkinsfile`. Save and click **Build Now**. If the work is later merged into `main`, switch the Jenkins branch to `*/main`. The Jenkinsfile also polls the Git repository every five minutes for new commits while Jenkins is running; a local Jenkins instance cannot receive a public GitHub webhook without an accessible endpoint.

The pipeline checks out the repository, builds the lightweight API image (`INSTALL_ML=false`), runs syntax checks and all pytest tests inside that image, then verifies the `/health` endpoint using FastAPI's test client. It does not download large research datasets or pretrained model weights and does not train a model. A green run proves the CI checks pass, not that scientific recommendations are valid or that the service is production-ready.

## Evidence for submission

After a successful run, capture screenshots showing (1) the Jenkins job and green stage view, (2) the console output with `pytest` summary and successful finish, and (3) the `Jenkinsfile` in the GitHub repository. If a run fails, fix the issue and capture the successful rerun; do not present a planned pipeline as an executed one. The assignment report should link to the actual pushed commit containing `Jenkinsfile`.

## Stop

```powershell
docker compose -f docker-compose.jenkins.yml down
```

This preserves Jenkins jobs in the named volume. Do not add `-v` unless you intend to delete that data.
