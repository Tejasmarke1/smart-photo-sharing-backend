# 🎯 COMPLETE DATABASE OVERVIEW

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "📊 KWIKPIC BACKEND - DATABASE SUMMARY" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 1. All Tables
Write-Host "1️⃣  ALL TABLES (10 total):" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "\dt"

# 2. Table Sizes
Write-Host "`n2️⃣  TABLE SIZES:" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"

# 3. Users Table
Write-Host "`n3️⃣  USERS TABLE:" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "\d users"

# 4. Albums Table
Write-Host "`n4️⃣  ALBUMS TABLE:" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "\d albums"

# 5. Photos Table
Write-Host "`n5️⃣  PHOTOS TABLE:" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "\d photos"

# 6. Faces Table (with vector embeddings)
Write-Host "`n6️⃣  FACES TABLE (Vector Embeddings):" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "\d faces"

# 7. Persons Table
Write-Host "`n7️⃣  PERSONS TABLE:" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "\d persons"

# 8. Face-Person Mapping
Write-Host "`n8️⃣  FACE_PERSON MAPPING:" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "\d face_person"

# 9. Payments Table
Write-Host "`n9️⃣  PAYMENTS TABLE:" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "\d payments"

# 10. Subscriptions Table
Write-Host "`n🔟 SUBSCRIPTIONS TABLE:" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "\d subscriptions"

# 11. Audit Logs Table
Write-Host "`n1️⃣1️⃣  AUDIT_LOGS TABLE:" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "\d audit_logs"

# 12. Downloads Table
Write-Host "`n1️⃣2️⃣  DOWNLOADS TABLE:" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "\d downloads"

# 13. All Foreign Key Relationships
Write-Host "`n1️⃣3️⃣  FOREIGN KEY RELATIONSHIPS:" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "
SELECT 
  tc.table_name AS from_table, 
  kcu.column_name AS from_column, 
  ccu.table_name AS to_table,
  ccu.column_name AS to_column,
  rc.delete_rule
FROM information_schema.table_constraints AS tc 
JOIN information_schema.key_column_usage AS kcu
  ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage AS ccu
  ON ccu.constraint_name = tc.constraint_name
JOIN information_schema.referential_constraints AS rc
  ON tc.constraint_name = rc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
ORDER BY tc.table_name, kcu.column_name;
"

# 14. All Indexes
Write-Host "`n1️⃣4️⃣  ALL INDEXES:" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "
SELECT 
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
"

# 15. All Enums
Write-Host "`n1️⃣5️⃣  ALL ENUMS:" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "
SELECT 
    t.typname AS enum_name,
    e.enumlabel AS enum_value
FROM pg_type t 
JOIN pg_enum e ON t.oid = e.enumtypid  
JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace
WHERE n.nspname = 'public'
ORDER BY t.typname, e.enumsortorder;
"

# 16. Database Extensions
Write-Host "`n1️⃣6️⃣  INSTALLED EXTENSIONS:" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "\dx"

# 17. Migration History
Write-Host "`n1️⃣7️⃣  MIGRATION HISTORY:" -ForegroundColor Green
poetry run alembic history --verbose

# 18. Current Migration Version
Write-Host "`n1️⃣8️⃣  CURRENT VERSION:" -ForegroundColor Green
poetry run alembic current

# 19. Database Statistics
Write-Host "`n1️⃣9️⃣  DATABASE STATISTICS:" -ForegroundColor Green
docker-compose exec postgres psql -U backend -d backend -c "
SELECT 
    schemaname,
    COUNT(*) AS table_count,
    SUM(n_tup_ins) AS total_inserts,
    SUM(n_tup_upd) AS total_updates,
    SUM(n_tup_del) AS total_deletes
FROM pg_stat_user_tables
WHERE schemaname = 'public'
GROUP BY schemaname;
"

# 20. Connection Info
Write-Host "`n2️⃣0️⃣  CONNECTION INFO:" -ForegroundColor Green
Write-Host "Database URL: postgresql://backend:backend@localhost:5432/backend" -ForegroundColor Yellow
Write-Host "pgAdmin Connection:" -ForegroundColor Yellow
Write-Host "  Host: localhost" -ForegroundColor White
Write-Host "  Port: 5432" -ForegroundColor White
Write-Host "  Database: backend" -ForegroundColor White
Write-Host "  Username: backend" -ForegroundColor White
Write-Host "  Password: backend" -ForegroundColor White

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "✅ DATABASE OVERVIEW COMPLETE!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

# Summary Table
Write-Host "📋 QUICK SUMMARY:" -ForegroundColor Cyan
Write-Host @"

Tables Created:
├── 👤 users              (Photographers, editors, guests, admins)
├── 📁 albums             (Events with sharing codes)
├── 📸 photos             (Image metadata & processing status)
├── 👁️  faces              (Detected faces with 512-dim embeddings)
├── 👥 persons            (Face clusters/labels)
├── 🔗 face_person        (Face-to-person mapping)
├── 💳 payments           (Razorpay transactions)
├── 📊 subscriptions      (User subscription plans)
├── 📝 audit_logs         (Compliance & tracking)
└── ⬇️  downloads          (Download tracking)

Key Features:
✅ pgvector extension for face recognition
✅ Foreign keys with cascading deletes
✅ Soft delete support (users, albums, photos, persons)
✅ Timestamp tracking (created_at, updated_at)
✅ Comprehensive indexing
✅ Enum types for status fields

"@ -ForegroundColor White

Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Create seed data: poetry run python scripts/dev/seed_data.py" -ForegroundColor White
Write-Host "  2. Build API endpoints: src/api/v1/endpoints/" -ForegroundColor White
Write-Host "  3. Implement authentication: src/core/security.py" -ForegroundColor White
Write-Host "  4. Setup face detection: src/services/face/" -ForegroundColor White
Write-Host "  5. Configure Celery workers: src/tasks/workers/" -ForegroundColor White

Write-Host "`n🚀 Ready for development!" -ForegroundColor Green