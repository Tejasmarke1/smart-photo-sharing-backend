resource "aws_db_subnet_group" "main" {
  name       = "--db-subnet"
  subnet_ids = var.subnet_ids

  tags = {
    Name = "--db-subnet"
  }
}

resource "aws_security_group" "rds" {
  name        = "--rds-sg"
  description = "Security group for RDS"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "--rds-sg"
  }
}

resource "aws_db_instance" "main" {
  identifier        = "-"
  engine            = "postgres"
  engine_version    = "16.1"
  instance_class    = var.instance_class
  allocated_storage = var.allocated_storage
  storage_encrypted = true
  
  db_name  = var.database_name
  username = var.master_username
  password = var.master_password
  
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  
  backup_retention_period = var.backup_retention_period
  backup_window          = "03:00-04:00"
  maintenance_window     = "mon:04:00-mon:05:00"
  
  skip_final_snapshot = var.environment != "production"
  
  tags = {
    Name        = "-"
    Environment = var.environment
  }
}
