import numpy as np
from scipy.spatial import distance as dist
from typing import List, Tuple, Optional

class CentroidTracker:
    """
    Simple centroid-based object tracker.
    Assigns persistent IDs to detections across frames.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3):
        self.next_id = 0
        self.objects = {}           # id -> (centroid_x, centroid_y)
        self.bboxes = {}            # id -> (x1, y1, x2, y2)
        self.disappeared = {}       # id -> frame count since last seen
        self.hits = {}              # id -> number of detections
        self.max_age = max_age
        self.min_hits = min_hits

    def update(self, detections: List[List[float]]) -> List[Tuple[int, Tuple[float, float], Tuple[float, float, float, float]]]:
        """
        Update tracker with new detections.
        :param detections: list of [x1, y1, x2, y2, confidence]
        :return: list of (object_id, centroid, bbox) for active objects
        """
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_age:
                    self._deregister(obj_id)
            return []

        # Compute centroids
        centroids = []
        bboxes = []
        for (x1, y1, x2, y2, _) in detections:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centroids.append((cx, cy))
            bboxes.append((x1, y1, x2, y2))

        # If no existing objects, register all new ones
        if len(self.objects) == 0:
            for i in range(len(centroids)):
                self._register(centroids[i], bboxes[i])
        else:
            # Match existing objects to new centroids using Hungarian-like greedy
            object_ids = list(self.objects.keys())
            object_centroids = np.array([self.objects[oid] for oid in object_ids])
            new_centroids = np.array(centroids)

            if len(object_centroids) > 0 and len(new_centroids) > 0:
                D = dist.cdist(object_centroids, new_centroids)
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                used_rows = set()
                used_cols = set()
                for row, col in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue
                    if D[row, col] < 100:  # distance threshold (pixels)
                        obj_id = object_ids[row]
                        self.objects[obj_id] = (new_centroids[col][0], new_centroids[col][1])
                        self.bboxes[obj_id] = bboxes[col]
                        self.disappeared[obj_id] = 0
                        self.hits[obj_id] += 1
                        used_rows.add(row)
                        used_cols.add(col)
                # Unmatched existing objects → disappeared
                for row, obj_id in enumerate(object_ids):
                    if row not in used_rows:
                        self.disappeared[obj_id] += 1
                        if self.disappeared[obj_id] > self.max_age:
                            self._deregister(obj_id)
                # Unmatched new centroids → new objects
                for col in range(len(centroids)):
                    if col not in used_cols:
                        self._register(centroids[col], bboxes[col])
            else:
                # No centroids but existing objects
                for obj_id in object_ids:
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_age:
                        self._deregister(obj_id)

        # Build active list (only objects with enough hits)
        active = []
        for obj_id in list(self.objects.keys()):
            if self.hits.get(obj_id, 0) >= self.min_hits:
                active.append((obj_id, self.objects[obj_id], self.bboxes[obj_id]))
        return active

    def _register(self, centroid: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> None:
        """Register a new object."""
        self.objects[self.next_id] = centroid
        self.bboxes[self.next_id] = bbox
        self.disappeared[self.next_id] = 0
        self.hits[self.next_id] = 1
        self.next_id += 1

    def _deregister(self, obj_id: int) -> None:
        """Remove an object from tracking."""
        self.objects.pop(obj_id, None)
        self.bboxes.pop(obj_id, None)
        self.disappeared.pop(obj_id, None)
        self.hits.pop(obj_id, None)