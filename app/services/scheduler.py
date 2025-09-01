"""
Scheduler service for managing periodic tasks.
"""

import asyncio
import schedule
import time
import threading
from datetime import datetime
from typing import Optional
from loguru import logger

from app.core.config import settings


class SchedulerService:
    """Service for managing scheduled tasks."""
    
    def __init__(self):
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start the scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        logger.info("Starting scheduler service...")
        
        # Schedule tasks
        self._schedule_tasks()
        
        # Start scheduler thread
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Scheduler service started successfully")
    
    def stop(self):
        """Stop the scheduler."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        logger.info("Stopping scheduler service...")
        self.is_running = False
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Scheduler service stopped")
    
    def _schedule_tasks(self):
        """Schedule periodic tasks."""
        # Log collection every 10 minutes
        schedule.every(settings.LOG_COLLECTION_INTERVAL).seconds.do(
            self._trigger_log_collection
        )
        
        # Model training daily at 2 AM
        schedule.every().day.at("02:00").do(
            self._trigger_model_training
        )
        
        # Latent space update every hour
        schedule.every().hour.do(
            self._trigger_latent_space_update
        )
        
        logger.info("Scheduled tasks configured")
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)  # Continue after error
    
    def _trigger_log_collection(self):
        """Trigger log collection task."""
        try:
            logger.info("Scheduled log collection triggered")
            # This would typically call the log collector service
            # For now, we'll just log the event
            asyncio.create_task(self._async_log_collection())
        except Exception as e:
            logger.error(f"Error triggering log collection: {e}")
    
    def _trigger_model_training(self):
        """Trigger model training task."""
        try:
            logger.info("Scheduled model training triggered")
            # This would typically call the model trainer service
            asyncio.create_task(self._async_model_training())
        except Exception as e:
            logger.error(f"Error triggering model training: {e}")
    
    def _trigger_latent_space_update(self):
        """Trigger latent space update task."""
        try:
            logger.info("Scheduled latent space update triggered")
            # This would typically call the latent space service
            asyncio.create_task(self._async_latent_space_update())
        except Exception as e:
            logger.error(f"Error triggering latent space update: {e}")
    
    async def _async_log_collection(self):
        """Async wrapper for log collection."""
        try:
            # Import here to avoid circular imports
            from main import log_collector
            logs = await log_collector.collect_system_logs()
            logger.info(f"Collected {len(logs)} logs via scheduler")
        except Exception as e:
            logger.error(f"Error in async log collection: {e}")
    
    async def _async_model_training(self):
        """Async wrapper for model training."""
        try:
            # Import here to avoid circular imports
            from main import log_collector, model_trainer
            
            # Get recent logs for training
            logs = await log_collector.get_recent_logs(minutes=1440)  # Last 24 hours
            
            if logs:
                result = await model_trainer.train_model(logs)
                if result.get("success"):
                    logger.info("Model training completed via scheduler")
                else:
                    logger.error(f"Model training failed: {result.get('message')}")
            else:
                logger.warning("No logs available for model training")
                
        except Exception as e:
            logger.error(f"Error in async model training: {e}")
    
    async def _async_latent_space_update(self):
        """Async wrapper for latent space update."""
        try:
            # Import here to avoid circular imports
            from main import log_collector, latent_space_service
            
            # Get recent logs for latent space update
            logs = await log_collector.get_recent_logs(minutes=60)  # Last hour
            
            if logs:
                await latent_space_service.update_latent_space(logs)
                logger.info("Latent space updated via scheduler")
            else:
                logger.info("No new logs for latent space update")
                
        except Exception as e:
            logger.error(f"Error in async latent space update: {e}")
    
    def get_scheduled_tasks(self):
        """Get information about scheduled tasks."""
        tasks = []
        for job in schedule.get_jobs():
            tasks.append({
                'function': job.job_func.__name__,
                'next_run': job.next_run.isoformat() if job.next_run else None,
                'interval': str(job.interval) if hasattr(job, 'interval') else 'N/A'
            })
        return tasks
