pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    triggers {
        pollSCM('H/5 * * * *')
    }

    environment {
        IMAGE_NAME = 'academic-recommendation-system:jenkins'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker image') {
            steps {
                sh 'docker build --build-arg INSTALL_ML=false -t "$IMAGE_NAME" .'
            }
        }

        stage('Syntax and unit tests') {
            steps {
                sh '''
                    docker run --rm --entrypoint sh "$IMAGE_NAME" -c '
                        python -m compileall -q app scripts tests &&
                        python -m pytest -q
                    '
                '''
            }
        }

        stage('Container smoke test') {
            steps {
                sh '''
                    docker run --rm --entrypoint python "$IMAGE_NAME" -c '
                        from fastapi.testclient import TestClient
                        from app.main import app
                        response = TestClient(app).get("/health")
                        assert response.status_code == 200, response.text
                        assert response.json()["status"] == "ok"
                    '
                '''
            }
        }
    }
}
