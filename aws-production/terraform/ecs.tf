# ECS Cluster
resource "aws_ecs_cluster" "mlflow" {
  name = "${var.project_name}-mlflow-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name        = "${var.project_name}-mlflow-cluster"
    Environment = var.environment
  }
}

# ECS Security Group
resource "aws_security_group" "ecs_tasks" {
  name_prefix = "${var.project_name}-ecs-tasks-"
  vpc_id      = aws_vpc.mlflow.id

  ingress {
    description     = "MLflow Server"
    from_port       = 5000
    to_port         = 5000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-ecs-tasks-sg"
    Environment = var.environment
  }
}

# ECS Task Execution Role
resource "aws_iam_role" "ecs_task_execution_role" {
  name = "${var.project_name}-ecs-task-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-ecs-task-execution-role"
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Additional policy for Secrets Manager
resource "aws_iam_role_policy" "ecs_secrets_policy" {
  name = "${var.project_name}-ecs-secrets-policy"
  role = aws_iam_role.ecs_task_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.db_credentials.arn
        ]
      }
    ]
  })
}

# ECS Task Role (for MLflow to access S3)
resource "aws_iam_role" "ecs_task_role" {
  name = "${var.project_name}-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-ecs-task-role"
    Environment = var.environment
  }
}

resource "aws_iam_role_policy" "ecs_s3_policy" {
  name = "${var.project_name}-ecs-s3-policy"
  role = aws_iam_role.ecs_task_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::fraud-models-test",
          "arn:aws:s3:::fraud-models-test/*"
        ]
      }
    ]
  })
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "mlflow" {
  name              = "/ecs/${var.project_name}-mlflow"
  retention_in_days = 7

  tags = {
    Name        = "${var.project_name}-mlflow-logs"
    Environment = var.environment
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "mlflow" {
  family                   = "${var.project_name}-mlflow"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name  = "mlflow"
      image = "public.ecr.aws/docker/library/python:3.11-slim"
      
      essential = true
      
      portMappings = [
        {
          containerPort = 5000
          protocol      = "tcp"
        }
      ]

      command = [
        "bash", "-c", 
        "apt-get update && apt-get install -y curl && pip install mlflow[extras]==2.18.0 boto3 psycopg2-binary bcrypt && echo $MLFLOW_AUTH_CONFIG | base64 -d > /tmp/mlflow-auth.ini && MLFLOW_AUTH_CONFIG_PATH=/tmp/mlflow-auth.ini mlflow server --backend-store-uri postgresql://$DB_USER:$DB_PASSWORD@$DB_ENDPOINT/$DB_NAME --default-artifact-root s3://fraud-models-test/mlflow-artifacts --host 0.0.0.0 --port 5000 --app-name basic-auth"
      ]

      environment = [
        {
          name  = "AWS_DEFAULT_REGION"
          value = var.aws_region
        },
        {
          name  = "MLFLOW_AUTH_CONFIG"
          value = "W21sZmxvd10KYWRtaW5fdXNlcm5hbWUgPSBhZG1pbgphZG1pbl9wYXNzd29yZCA9IG1sZmxvdy1hZG1pbi0yMDI1CmRlZmF1bHRfcGVybWlzc2lvbiA9IE5PX1BFUk1JU1NJT05TCmRhdGFiYXNlX3VyaSA9IHNxbGl0ZTovLy8vdG1wL21sZmxvdy1hdXRoLmRi"
        }
      ]

      secrets = [
        {
          name      = "DB_USER"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:username::"
        },
        {
          name      = "DB_PASSWORD" 
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:password::"
        },
        {
          name      = "DB_ENDPOINT"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:endpoint::"
        },
        {
          name      = "DB_NAME"
          valueFrom = "${aws_secretsmanager_secret.db_credentials.arn}:dbname::"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.mlflow.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:5000/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = {
    Name        = "${var.project_name}-mlflow-task"
    Environment = var.environment
  }
}

# ECS Service
resource "aws_ecs_service" "mlflow" {
  name            = "${var.project_name}-mlflow"
  cluster         = aws_ecs_cluster.mlflow.id
  task_definition = aws_ecs_task_definition.mlflow.arn
  launch_type     = "FARGATE"
  desired_count   = 2

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets          = aws_subnet.private[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.mlflow.arn
    container_name   = "mlflow"
    container_port   = 5000
  }

  depends_on = [aws_lb_listener.mlflow]

  tags = {
    Name        = "${var.project_name}-mlflow-service"
    Environment = var.environment
  }
}