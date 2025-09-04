# RDS Subnet Group
resource "aws_db_subnet_group" "mlflow" {
  name       = "${var.project_name}-mlflow-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id

  tags = {
    Name        = "${var.project_name}-mlflow-db-subnet-group"
    Environment = var.environment
  }
}

# RDS Security Group
resource "aws_security_group" "rds" {
  name_prefix = "${var.project_name}-rds-"
  vpc_id      = aws_vpc.mlflow.id

  ingress {
    description     = "PostgreSQL"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-rds-sg"
    Environment = var.environment
  }
}

# RDS Instance
resource "aws_db_instance" "mlflow" {
  identifier     = "${var.project_name}-mlflow-db"
  engine         = "postgres"
  engine_version = "15.7"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true

  db_name  = "mlflow_db"
  username = "mlflow_user"
  password = random_password.db_password.result

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.mlflow.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = true
  deletion_protection = false

  tags = {
    Name        = "${var.project_name}-mlflow-db"
    Environment = var.environment
  }
}

# Random password for RDS
resource "random_password" "db_password" {
  length  = 16
  special = true
}

# Store DB credentials in AWS Secrets Manager
resource "aws_secretsmanager_secret" "db_credentials" {
  name = "${var.project_name}-mlflow-db-credentials"
  
  tags = {
    Name        = "${var.project_name}-mlflow-db-credentials"
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = aws_db_instance.mlflow.username
    password = random_password.db_password.result
    endpoint = aws_db_instance.mlflow.endpoint
    port     = aws_db_instance.mlflow.port
    dbname   = aws_db_instance.mlflow.db_name
  })
}