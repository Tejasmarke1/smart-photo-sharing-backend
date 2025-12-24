"""S3 storage service for photo uploads and downloads."""
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, BotoCoreError
from typing import Optional, Dict, List, Tuple, Any
import secrets
import logging
from datetime import datetime

from src.app.config import settings

logger = logging.getLogger(__name__)


class S3ServiceError(Exception):
    """Custom exception for S3 service errors."""
    pass


class S3Service:
    """Service for S3 operations with comprehensive error handling."""
    
    def __init__(self):
        """Initialize S3 client with configuration."""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.S3_REGION,
                config=Config(
                    signature_version='s3v4',
                    s3={'addressing_style': 'virtual'}
                )
            )
            self.bucket_name = settings.S3_BUCKET_NAME
            logger.info(f"S3 Service initialized for bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise S3ServiceError(f"S3 initialization failed: {str(e)}")
    
    def generate_s3_key(self, album_id: str, filename: str, prefix: str = "photos") -> str:
        """
        Generate unique S3 key for photo.
        
        Args:
            album_id: Album UUID string
            filename: Original filename
            prefix: Subfolder name (default: photos)
            
        Returns:
            S3 key path: albums/{album_id}/{prefix}/{random_token}.{ext}
        """
        # Extract file extension
        file_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'jpg'
        
        # Generate random token for uniqueness
        token = secrets.token_urlsafe(16)
        
        # Format: albums/{album_id}/{prefix}/{random_token}.{ext}
        return f"albums/{album_id}/{prefix}/{token}.{file_ext}"
    
    def generate_thumbnail_key(self, original_key: str, size: str) -> str:
        """
        Generate S3 key for thumbnail.
        
        Args:
            original_key: Original photo S3 key
            size: Thumbnail size (small, medium, large)
            
        Returns:
            Thumbnail S3 key
        """
        # albums/xxx/photos/yyy.jpg -> albums/xxx/thumbnails/small_yyy.jpg
        parts = original_key.rsplit('/', 1)
        directory = parts[0].replace('/photos', '/thumbnails')
        filename = parts[1]
        return f"{directory}/{size}_{filename}"
    
    def generate_presigned_upload_url(
        self,
        s3_key: str,
        content_type: str,
        expires_in: int = 3600,
        use_multipart: bool = False,
        max_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate presigned URL for direct S3 upload.
        
        Args:
            s3_key: S3 object key
            content_type: MIME type
            expires_in: URL expiration in seconds (default: 1 hour)
            use_multipart: Use POST with form fields (for browser uploads)
            max_size: Maximum file size in bytes (for POST uploads)
            
        Returns:
            Dict with upload_url and optional fields
            
        Raises:
            S3ServiceError: If URL generation fails
        """
        try:
            if use_multipart:
                # Generate presigned POST (for browser form uploads)
                conditions = [
                    {'Content-Type': content_type},
                    ['content-length-range', 1, max_size or 100 * 1024 * 1024]
                ]
                
                response = self.s3_client.generate_presigned_post(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Fields={'Content-Type': content_type},
                    Conditions=conditions,
                    ExpiresIn=expires_in
                )
                
                logger.debug(f"Generated POST presigned URL for: {s3_key}")
                return {
                    'upload_url': response['url'],
                    'fields': response['fields'],
                    'method': 'POST'
                }
            else:
                # Generate presigned PUT (for direct uploads)
                upload_url = self.s3_client.generate_presigned_url(
                    'put_object',
                    Params={
                        'Bucket': self.bucket_name,
                        'Key': s3_key,
                        'ContentType': content_type
                    },
                    ExpiresIn=expires_in
                )
                
                logger.debug(f"Generated PUT presigned URL for: {s3_key}")
                return {
                    'upload_url': upload_url,
                    'method': 'PUT'
                }
                
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise S3ServiceError(f"Failed to generate upload URL: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error generating presigned URL: {e}")
            raise S3ServiceError(f"Unexpected error: {str(e)}")
    
    def generate_presigned_download_url(
        self,
        s3_key: str,
        expires_in: int = 3600,
        filename: Optional[str] = None,
        inline: bool = False
    ) -> str:
        """
        Generate presigned URL for download.
        
        Args:
            s3_key: S3 object key
            expires_in: URL expiration in seconds
            filename: Optional filename for Content-Disposition header
            inline: If True, display in browser; if False, force download
            
        Returns:
            Presigned download URL
            
        Raises:
            S3ServiceError: If URL generation fails
        """
        try:
            params = {
                'Bucket': self.bucket_name,
                'Key': s3_key
            }
            
            if filename:
                disposition = f'inline; filename="{filename}"' if inline else f'attachment; filename="{filename}"'
                params['ResponseContentDisposition'] = disposition
            
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params=params,
                ExpiresIn=expires_in
            )
            
            logger.debug(f"Generated download URL for: {s3_key}")
            return url
            
        except ClientError as e:
            logger.error(f"Error generating download URL: {e}")
            raise S3ServiceError(f"Failed to generate download URL: {str(e)}")
    
    def initiate_multipart_upload(
        self,
        s3_key: str,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Initiate multipart upload for large files.
        
        Args:
            s3_key: S3 object key
            content_type: MIME type
            metadata: Optional metadata dict
            
        Returns:
            Upload ID string
            
        Raises:
            S3ServiceError: If initialization fails
        """
        try:
            params = {
                'Bucket': self.bucket_name,
                'Key': s3_key,
                'ContentType': content_type
            }
            
            if metadata:
                params['Metadata'] = metadata
            
            response = self.s3_client.create_multipart_upload(**params)
            upload_id = response['UploadId']
            
            logger.info(f"Initiated multipart upload: {s3_key} (ID: {upload_id})")
            return upload_id
            
        except ClientError as e:
            logger.error(f"Error initiating multipart upload: {e}")
            raise S3ServiceError(f"Failed to initiate multipart upload: {str(e)}")
    
    def generate_multipart_presigned_urls(
        self,
        s3_key: str,
        upload_id: str,
        part_numbers: List[int],
        expires_in: int = 3600
    ) -> Dict[int, str]:
        """
        Generate presigned URLs for multipart upload parts.
        
        Args:
            s3_key: S3 object key
            upload_id: Multipart upload ID
            part_numbers: List of part numbers (1-indexed)
            expires_in: URL expiration in seconds
            
        Returns:
            Dict mapping part_number to presigned URL
            
        Raises:
            S3ServiceError: If URL generation fails
        """
        try:
            urls = {}
            for part_number in part_numbers:
                if part_number < 1 or part_number > 10000:
                    raise ValueError(f"Part number must be between 1 and 10000, got {part_number}")
                
                url = self.s3_client.generate_presigned_url(
                    'upload_part',
                    Params={
                        'Bucket': self.bucket_name,
                        'Key': s3_key,
                        'UploadId': upload_id,
                        'PartNumber': part_number
                    },
                    ExpiresIn=expires_in
                )
                urls[part_number] = url
            
            logger.debug(f"Generated {len(urls)} part URLs for: {s3_key}")
            return urls
            
        except (ClientError, ValueError) as e:
            logger.error(f"Error generating multipart URLs: {e}")
            raise S3ServiceError(f"Failed to generate part URLs: {str(e)}")
    
    def complete_multipart_upload(
        self,
        s3_key: str,
        upload_id: str,
        parts: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Complete multipart upload.
        
        Args:
            s3_key: S3 object key
            upload_id: Multipart upload ID
            parts: List of dicts with PartNumber and ETag
                   Example: [{'PartNumber': 1, 'ETag': '"abc123"'}, ...]
            
        Returns:
            Dict with Location, Bucket, Key, ETag
            
        Raises:
            S3ServiceError: If completion fails
        """
        try:
            # Validate parts
            if not parts:
                raise ValueError("Parts list cannot be empty")
            
            # Sort parts by PartNumber
            sorted_parts = sorted(parts, key=lambda x: x['PartNumber'])
            
            response = self.s3_client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': sorted_parts}
            )
            
            result = {
                'location': response.get('Location', ''),
                'bucket': response['Bucket'],
                'key': response['Key'],
                'etag': response['ETag']
            }
            
            logger.info(f"Completed multipart upload: {s3_key} (ID: {upload_id})")
            return result
            
        except (ClientError, ValueError) as e:
            logger.error(f"Error completing multipart upload: {e}")
            raise S3ServiceError(f"Failed to complete multipart upload: {str(e)}")
    
    def abort_multipart_upload(
        self,
        s3_key: str,
        upload_id: str
    ) -> None:
        """
        Abort multipart upload and cleanup parts.
        
        Args:
            s3_key: S3 object key
            upload_id: Multipart upload ID
            
        Raises:
            S3ServiceError: If abort fails
        """
        try:
            self.s3_client.abort_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                UploadId=upload_id
            )
            logger.info(f"Aborted multipart upload: {s3_key} (ID: {upload_id})")
            
        except ClientError as e:
            # Don't raise if upload doesn't exist (already completed/aborted)
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchUpload':
                logger.warning(f"Multipart upload not found: {upload_id}")
                return
            
            logger.error(f"Error aborting multipart upload: {e}")
            raise S3ServiceError(f"Failed to abort multipart upload: {str(e)}")
    
    def list_multipart_parts(
        self,
        s3_key: str,
        upload_id: str,
        max_parts: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        List uploaded parts for a multipart upload.
        
        Args:
            s3_key: S3 object key
            upload_id: Multipart upload ID
            max_parts: Maximum parts to list
            
        Returns:
            List of part dicts with PartNumber, Size, ETag, LastModified
        """
        try:
            response = self.s3_client.list_parts(
                Bucket=self.bucket_name,
                Key=s3_key,
                UploadId=upload_id,
                MaxParts=max_parts
            )
            
            parts = response.get('Parts', [])
            logger.debug(f"Listed {len(parts)} parts for: {s3_key}")
            return parts
            
        except ClientError as e:
            logger.error(f"Error listing multipart parts: {e}")
            raise S3ServiceError(f"Failed to list parts: {str(e)}")
    
    def delete_object(self, s3_key: str) -> None:
        """
        Delete object from S3.
        
        Args:
            s3_key: S3 object key
            
        Raises:
            S3ServiceError: If deletion fails
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            logger.info(f"Deleted S3 object: {s3_key}")
            
        except ClientError as e:
            logger.error(f"Error deleting object: {e}")
            raise S3ServiceError(f"Failed to delete object: {str(e)}")
    
    def delete_objects_bulk(self, s3_keys: List[str]) -> Tuple[int, List[str]]:
        """
        Delete multiple objects from S3 (up to 1000 per call).
        
        Args:
            s3_keys: List of S3 object keys
            
        Returns:
            Tuple of (success_count, failed_keys)
            
        Raises:
            S3ServiceError: If bulk delete fails
        """
        if not s3_keys:
            return 0, []
        
        if len(s3_keys) > 1000:
            logger.warning(f"Truncating bulk delete to 1000 objects (got {len(s3_keys)})")
            s3_keys = s3_keys[:1000]
        
        try:
            objects = [{'Key': key} for key in s3_keys]
            response = self.s3_client.delete_objects(
                Bucket=self.bucket_name,
                Delete={'Objects': objects, 'Quiet': False}
            )
            
            deleted = response.get('Deleted', [])
            errors = response.get('Errors', [])
            
            success_count = len(deleted)
            failed_keys = [error['Key'] for error in errors]
            
            logger.info(f"Bulk delete: {success_count} succeeded, {len(failed_keys)} failed")
            
            if errors:
                for error in errors:
                    logger.error(f"Failed to delete {error['Key']}: {error.get('Message', 'Unknown error')}")
            
            return success_count, failed_keys
            
        except ClientError as e:
            logger.error(f"Error in bulk delete: {e}")
            raise S3ServiceError(f"Bulk delete failed: {str(e)}")
    
    def check_object_exists(self, s3_key: str) -> bool:
        """
        Check if object exists in S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if exists, False otherwise
        """
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                return False
            logger.error(f"Error checking object existence: {e}")
            raise S3ServiceError(f"Failed to check object: {str(e)}")
    
    def get_object_metadata(self, s3_key: str) -> Dict[str, Any]:
        """
        Get object metadata from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Dict with content_length, content_type, etag, last_modified, metadata
            
        Raises:
            S3ServiceError: If metadata retrieval fails
        """
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return {
                'content_length': response['ContentLength'],
                'content_type': response.get('ContentType', 'application/octet-stream'),
                'etag': response['ETag'].strip('"'),
                'last_modified': response['LastModified'],
                'metadata': response.get('Metadata', {})
            }
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                raise S3ServiceError(f"Object not found: {s3_key}")
            logger.error(f"Error getting object metadata: {e}")
            raise S3ServiceError(f"Failed to get metadata: {str(e)}")
    
    def copy_object(
        self,
        source_key: str,
        destination_key: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None
    ) -> str:
        """
        Copy object within S3.
        
        Args:
            source_key: Source S3 key
            destination_key: Destination S3 key
            metadata: Optional metadata to set
            content_type: Optional content type override
            
        Returns:
            ETag of copied object
            
        Raises:
            S3ServiceError: If copy fails
        """
        try:
            copy_source = {
                'Bucket': self.bucket_name,
                'Key': source_key
            }
            
            params = {
                'Bucket': self.bucket_name,
                'CopySource': copy_source,
                'Key': destination_key
            }
            
            if metadata:
                params['Metadata'] = metadata
                params['MetadataDirective'] = 'REPLACE'
            
            if content_type:
                params['ContentType'] = content_type
            
            response = self.s3_client.copy_object(**params)
            etag = response['CopyObjectResult']['ETag'].strip('"')
            
            logger.info(f"Copied object: {source_key} -> {destination_key}")
            return etag
            
        except ClientError as e:
            logger.error(f"Error copying object: {e}")
            raise S3ServiceError(f"Failed to copy object: {str(e)}")
    
    def list_objects_by_prefix(
        self,
        prefix: str,
        max_keys: int = 1000,
        delimiter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List objects with given prefix.
        
        Args:
            prefix: S3 key prefix
            max_keys: Maximum keys to return (max 1000)
            delimiter: Delimiter for grouping (e.g., '/' for folders)
            
        Returns:
            List of object dicts with key, size, last_modified, etag
            
        Raises:
            S3ServiceError: If listing fails
        """
        try:
            params = {
                'Bucket': self.bucket_name,
                'Prefix': prefix,
                'MaxKeys': min(max_keys, 1000)
            }
            
            if delimiter:
                params['Delimiter'] = delimiter
            
            response = self.s3_client.list_objects_v2(**params)
            
            if 'Contents' not in response:
                return []
            
            objects = [
                {
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag'].strip('"')
                }
                for obj in response['Contents']
            ]
            
            logger.debug(f"Listed {len(objects)} objects with prefix: {prefix}")
            return objects
            
        except ClientError as e:
            logger.error(f"Error listing objects: {e}")
            raise S3ServiceError(f"Failed to list objects: {str(e)}")
    
    def calculate_multipart_info(
        self,
        filesize: int,
        part_size: int = 5 * 1024 * 1024
    ) -> Tuple[int, int]:
        """
        Calculate number of parts for multipart upload.
        
        Args:
            filesize: Total file size in bytes
            part_size: Desired size of each part (min 5MB)
            
        Returns:
            Tuple of (total_parts, actual_part_size)
            
        Note:
            AWS limits: min 5MB per part (except last), max 10,000 parts
        """
        # AWS requirements
        min_part_size = 5 * 1024 * 1024  # 5MB
        max_parts = 10000
        
        # Ensure part size is at least 5MB
        part_size = max(part_size, min_part_size)
        
        # Calculate total parts
        total_parts = (filesize + part_size - 1) // part_size
        
        # If too many parts, increase part size
        if total_parts > max_parts:
            part_size = (filesize + max_parts - 1) // max_parts
            # Round up to nearest MB
            part_size = ((part_size + 1024 * 1024 - 1) // (1024 * 1024)) * (1024 * 1024)
            total_parts = (filesize + part_size - 1) // part_size
        
        logger.debug(f"Multipart calculation: {filesize} bytes = {total_parts} parts of {part_size} bytes")
        return total_parts, part_size
    
    def get_bucket_info(self) -> Dict[str, Any]:
        """
        Get bucket information and verify access.
        
        Returns:
            Dict with bucket name, region, creation_date
            
        Raises:
            S3ServiceError: If bucket access fails
        """
        try:
            # Check bucket exists and we have access
            response = self.s3_client.head_bucket(Bucket=self.bucket_name)
            
            # Get bucket location
            location_response = self.s3_client.get_bucket_location(Bucket=self.bucket_name)
            region = location_response['LocationConstraint'] or 'us-east-1'
            
            return {
                'bucket_name': self.bucket_name,
                'region': region,
                'accessible': True
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '403':
                raise S3ServiceError(f"Access denied to bucket: {self.bucket_name}")
            elif error_code == '404':
                raise S3ServiceError(f"Bucket not found: {self.bucket_name}")
            
            logger.error(f"Error getting bucket info: {e}")
            raise S3ServiceError(f"Failed to access bucket: {str(e)}")

    def download_file(self, s3_key: str) -> bytes:
        """
        Download file from S3 and return raw bytes.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            File content as bytes
            
        Raises:
            S3ServiceError: If download fails
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            file_bytes = response['Body'].read()
            logger.debug(f"Downloaded {len(file_bytes)} bytes from: {s3_key}")
            return file_bytes
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchKey':
                raise S3ServiceError(f"Object not found: {s3_key}")
            logger.error(f"Error downloading file: {e}")
            raise S3ServiceError(f"Failed to download file: {str(e)}")

    def upload_file(
        self,
        file_data: bytes,
        s3_key: str,
        content_type: str = 'application/octet-stream',
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Upload file bytes directly to S3.
        
        Args:
            file_data: File content as bytes
            s3_key: S3 object key
            content_type: MIME type
            metadata: Optional metadata dict
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            S3ServiceError: If upload fails
        """
        try:
            params = {
                'Bucket': self.bucket_name,
                'Key': s3_key,
                'Body': file_data,
                'ContentType': content_type
            }
            
            if metadata:
                params['Metadata'] = metadata
            
            self.s3_client.put_object(**params)
            logger.info(f"Uploaded {len(file_data)} bytes to: {s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error uploading file: {e}")
            raise S3ServiceError(f"Failed to upload file: {str(e)}")

    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3 (alias for delete_object for consistency).
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if successful
            
        Raises:
            S3ServiceError: If deletion fails
        """
        self.delete_object(s3_key)
        return True