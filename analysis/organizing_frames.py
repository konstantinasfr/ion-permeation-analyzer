import numpy as np


def cluster_frames_by_closest_residue(distance_data):
    clustered_results = {}

    for ion_id, frame_list in distance_data.items():
        clusters = []
        prev_residue = None
        start_frame = None
        distances = []

        for frame_data in frame_list:
            frame = frame_data["frame"]
            residues = frame_data["residues"]

            # Find the closest residue (key with smallest value)
            closest_residue, closest_distance = min(residues.items(), key=lambda item: item[1])

            if closest_residue != prev_residue:
                # If ending a previous cluster, store it
                if prev_residue is not None:
                    clusters.append({
                        "residue": prev_residue,
                        "start": start_frame,
                        "end": prev_frame,
                        "frames": prev_frame - start_frame + 1,
                        "mean_distance": sum(distances) / len(distances)
                    })
                # Start a new cluster
                start_frame = frame
                prev_residue = closest_residue
                distances = [closest_distance]
            else:
                distances.append(closest_distance)

            prev_frame = frame

        # Add final cluster
        if prev_residue is not None:
            clusters.append({
                "residue": prev_residue,
                "start": start_frame,
                "end": prev_frame,
                "frames": prev_frame - start_frame + 1,
                "mean_distance": sum(distances) / len(distances)
            })

        clustered_results[ion_id] = clusters

    return clustered_results
