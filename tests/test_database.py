"""
Unit tests for database models, connection, and data persistence.
"""

from src.database.connection import get_session, init_schema
from src.database.models import ResumeRecord, JobRecord

def test_database_schema_and_query():
    assert init_schema() is True
    session = get_session()

    # Query jobs
    jobs_count = session.query(JobRecord).count()
    assert jobs_count > 0, "Jobs should be seeded in the database"

    # Insert test resume record
    test_resume = ResumeRecord(
        filename="test_candidate.pdf",
        candidate_name="Test Candidate",
        email="test.candidate@test.com",
        experience_years=3.0,
        raw_text="Test raw resume",
        cleaned_text="test raw resume"
    )
    session.add(test_resume)
    session.commit()

    # Query back
    retrieved = session.query(ResumeRecord).filter_by(email="test.candidate@test.com").first()
    assert retrieved is not None
    assert retrieved.candidate_name == "Test Candidate"

    # Cleanup test record
    session.delete(retrieved)
    session.commit()
    session.close()
