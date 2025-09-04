# Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.mlflow.id
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.mlflow.endpoint
  sensitive   = true
}

output "rds_port" {
  description = "RDS instance port"
  value       = aws_db_instance.mlflow.port
}

output "load_balancer_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.mlflow.dns_name
}

output "load_balancer_zone_id" {
  description = "Zone ID of the load balancer"
  value       = aws_lb.mlflow.zone_id
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.mlflow.name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.mlflow.name
}

output "secrets_manager_secret_arn" {
  description = "ARN of the Secrets Manager secret containing DB credentials"
  value       = aws_secretsmanager_secret.db_credentials.arn
}

output "mlflow_url" {
  description = "URL to access MLflow"
  value       = "http://${aws_lb.mlflow.dns_name}"
}