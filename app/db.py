from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

Base = declarative_base()


class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    camera_id = Column(String, default="default")
    tracker_id = Column(Integer)
    violation_type = Column(String)
    confidence = Column(Float)
    plate_number = Column(String, nullable=True)
    plate_confidence = Column(Float, nullable=True)
    evidence_image_path = Column(String)


engine = create_engine("sqlite:///violations.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


def log_violation(camera_id, tracker_id, violation_type, confidence,
                   plate_number, plate_confidence, evidence_image_path):
    session = Session()
    v = Violation(
        camera_id=camera_id,
        tracker_id=tracker_id,
        violation_type=violation_type,
        confidence=confidence,
        plate_number=plate_number,
        plate_confidence=plate_confidence,
        evidence_image_path=evidence_image_path
    )
    session.add(v)
    session.commit()
    session.close()