variable "project_name" {
  default = "kwikpic"
}

variable "environment" {
  default = "dev"
}

variable "aws_region" {
  default = "ap-south-1"
}

variable "vpc_cidr" {
  default = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  default = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  default = ["10.0.10.0/24", "10.0.11.0/24"]
}

variable "availability_zones" {
  default = ["ap-south-1a", "ap-south-1b"]
}

variable "database_name" {
  default = "kwikpic_dev"
}

variable "db_username" {
  sensitive = true
}

variable "db_password" {
  sensitive = true
}
