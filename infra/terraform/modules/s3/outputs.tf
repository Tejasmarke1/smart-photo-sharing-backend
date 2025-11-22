output "bucket_name" {
  value = aws_s3_bucket.photos.id
}

output "bucket_arn" {
  value = aws_s3_bucket.photos.arn
}
