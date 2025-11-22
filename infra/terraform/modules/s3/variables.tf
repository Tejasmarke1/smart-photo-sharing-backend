variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "allowed_origins" {
  type    = list(string)
  default = ["*"]
}
