"""
REST API Routes
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional
import time

from .models import (
    CameraInfo,
    CameraStats,
    PersonInfo,
    SystemStats,
    HealthResponse
)


def create_rest_router(manager, ws_manager, start_time):
    """Create REST API router with manager and websocket manager"""
    router = APIRouter()
    
    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            cameras=manager.get_camera_ids(),
            uptime_seconds=time.time() - start_time
        )
    
    @router.get("/api/cameras", response_model=List[CameraInfo])
    async def list_cameras():
        """List all configured cameras"""
        cameras = []
        for cam in manager.config.get_enabled_cameras():
            cameras.append(CameraInfo(
                id=cam['id'],
                name=cam['name'],
                rtsp_url=cam['rtsp_url'],
                enabled=cam.get('enabled', True)
            ))
        return cameras
    
    @router.get("/api/cameras/{camera_id}", response_model=CameraInfo)
    async def get_camera(camera_id: str):
        """Get specific camera information"""
        cam = manager.config.get_camera_by_id(camera_id)
        if not cam:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        return CameraInfo(
            id=cam['id'],
            name=cam['name'],
            rtsp_url=cam['rtsp_url'],
            enabled=cam.get('enabled', True)
        )
    
    @router.get("/api/cameras/{camera_id}/stats", response_model=CameraStats)
    async def get_camera_stats(camera_id: str):
        """Get camera statistics"""
        pipeline = manager.get_pipeline(camera_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        reader_stats = manager.readers[camera_id].get_stats()
        pipeline_stats = pipeline.get_stats()
        
        return CameraStats(
            camera_id=camera_id,
            running=reader_stats['running'],
            frame_count=reader_stats['frame_count'],
            fps=reader_stats['fps'],
            queue_size=reader_stats['queue_size'],
            frames_processed=pipeline_stats['frames_processed'],
            detections=pipeline_stats['detections'],
            tracks_active=pipeline_stats['tracks_active'],
            persons_identified=pipeline_stats['persons_identified']
        )
    
    @router.get("/api/persons", response_model=List[PersonInfo])
    async def list_persons():
        """List all identified persons"""
        persons = []
        for person_id, person in manager.person_db.persons.items():
            persons.append(PersonInfo(
                person_id=person.person_id,
                first_seen=person.first_seen,
                last_seen=person.last_seen,
                cameras_seen=person.cameras_seen,
                total_appearances=person.total_appearances,
                feature_count=len(person.feature_gallery)
            ))
        return persons
    
    @router.get("/api/persons/{person_id}", response_model=PersonInfo)
    async def get_person(person_id: int):
        """Get specific person information"""
        person = manager.person_db.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        return PersonInfo(
            person_id=person.person_id,
            first_seen=person.first_seen,
            last_seen=person.last_seen,
            cameras_seen=person.cameras_seen,
            total_appearances=person.total_appearances,
            feature_count=len(person.feature_gallery)
        )
    
    @router.get("/api/stats", response_model=SystemStats)
    async def get_system_stats():
        """Get system-wide statistics"""
        stats = manager.get_stats()
        
        return SystemStats(
            total_cameras=len(manager.get_camera_ids()),
            cameras_running=sum(
                1 for r in manager.readers.values() if r.is_alive()
            ),
            total_persons=stats['person_db']['total_persons'],
            total_appearances=stats['person_db']['total_appearances'],
            cameras=stats['cameras']
        )
    
    @router.post("/api/persons/{person_id}/reset")
    async def reset_person(person_id: int):
        """Reset a person's data (delete from database)"""
        if person_id not in manager.person_db.persons:
            raise HTTPException(status_code=404, detail="Person not found")
        
        del manager.person_db.persons[person_id]
        manager.person_db.save()
        
        return {"message": f"Person {person_id} deleted"}
    
    @router.post("/api/database/save")
    async def save_database():
        """Manually save person database to disk"""
        manager.person_db.save()
        return {"message": "Database saved successfully"}
    
    return router
