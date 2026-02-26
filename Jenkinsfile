pipeline {
    agent {
    kubernetes {
      yaml '''
        apiVersion: v1
        kind: Pod
        metadata:
          name: jenkins-agent
          namespace: devops
        spec:
          nodeSelector:
            kubernetes.io/hostname: drax
          containers:
          - name: docker
            image: docker:latest
            command:
            - cat
            tty: true
            volumeMounts:
             - mountPath: /var/run/docker.sock
               name: docker-sock
             - mountPath: /etc/docker/daemon.json
               name: daemon-config
               subPath: daemon.json
            env:
             - name: JENKINS_URL
               value: "http://jenkins-service.devops.svc.cluster.local:8080"
          - name: jnlp
            image: jenkins/inbound-agent:latest
            args: ["$(JENKINS_SECRET)", "$(JENKINS_NAME)"]
            env:
             - name: JENKINS_URL
               value: "http://jenkins-service.devops.svc.cluster.local:8080"
             - name: JENKINS_AGENT_PROTOCOLS
               value: "JNLP4-connect"
             - name: JENKINS_TUNNEL
               value: "jenkins-service.devops.svc.cluster.local:50000"
          volumes:
          - name: docker-sock
            hostPath:
              path: /var/run/docker.sock
          - name: daemon-config
            configMap:
              name: daemon-config
        '''
    }
  }

     triggers {
        githubPush()
      }

    environment {
        BUILD_VERSION = new Date().format('yyyy-MM-dd')
        DOCKER_IMAGE_TAG = "file-receiver-${env.BRANCH_NAME}:${env.BUILD_VERSION}"
        NEXUS_CRED = credentials('nexus-cred')
        NEXUS_DOCKER_REPO = '172.16.10.4:102'
    }

    stages {

       stage('Build docker image') {
            steps {
                container('docker') {
          sh "docker build -t ${env.NEXUS_DOCKER_REPO}/${env.DOCKER_IMAGE_TAG} ."
        }
            }
        }

        stage('Push docker image to nexus') {
            steps {
                container('docker') { // Dodano kontener docker
            withCredentials([usernamePassword(credentialsId: 'nexus-cred', usernameVariable: 'USER', passwordVariable: 'PASS')]) {
            sh "docker info"
             sh "cat /etc/docker/daemon.json"
                sh '''
                    PASSWORD=$(echo "$PASS" | tr -d '\\n')
                    echo "$PASSWORD" | docker login -u "$USER" --password-stdin "$NEXUS_DOCKER_REPO"
                    docker push "$NEXUS_DOCKER_REPO/$DOCKER_IMAGE_TAG"
                '''
            }
        }
            }
        }

    }
}