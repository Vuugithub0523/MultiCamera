import os
import cv2
import yaml
import argparse
import numpy as np
import time
from datetime import datetime
from scipy.spatial import distance
from tqdm.autonotebook import tqdm
from itertools import count as while_true
from object_detection import ObjectDetection
from feature_extraction import FeatureExtraction
from person_lifecycle_manager import PersonLifecycleManager, PersonState
from helpers import (
    map_bbox_letterbox,
    resize_with_letterbox,
    setup_resolution,
    stack_images,
)
from rtsp_multicam_loader import RTSPStreamLoader


def main(cfg):
    # ==================== KHỞI TẠO LIFECYCLE MANAGER ====================
    lifecycle_manager = PersonLifecycleManager(
        output_dir=cfg.get("tracking_log_dir", "./tracking_logs")
    )
    print("Đã khởi tạo Person Lifecycle Manager")
    
    # Cấu hình lifecycle
    lifecycle_manager.max_lost_frames = cfg.get("max_lost_frames", 30)
    lifecycle_manager.max_confirm_lost_frames = cfg.get("max_confirm_lost_frames", 90)
    lifecycle_manager.archive_after_seconds = cfg.get("archive_after_seconds", 300)
    
    # ==================== CẤU HÌNH TIME WINDOW ====================
    time_window_seconds = cfg.get("time_window_seconds", 3.0)
    print(f" Time window matching: {time_window_seconds} seconds")
    
    # ==================== CẤU HÌNH CAMERA TOPOLOGY ====================
    camera_topology = cfg.get("camera_topology", {})
    camera_transition_max_time = cfg.get("camera_transition_max_time", {})
    
    if camera_topology:
        print(f" Camera topology enabled:")
        for cam, connected in camera_topology.items():
            print(f"   Camera {cam} → {connected}")
        print(f"  Transition times: {len(camera_transition_max_time)} rules")
    else:
        print("  Camera topology not configured - using time window only")
    # ================================================================
    
    # Variable for save detected person features
    detected_persons = {}

    # Init object detecion
    object_detection = ObjectDetection(
        confidence_threshold=cfg["object_detection_threshold"],
        onnx_path=cfg["object_detection_model_path"],
        coco_names_path=cfg["object_detection_classes_path"],
        device=cfg["inference_model_device"],
    )

    # Init feature extraction
    feature_extraction = FeatureExtraction(
        onnx_path=cfg["feature_extraction_model_path"],
        device=cfg["inference_model_device"],
    )

    # Setup camera
    cam = {}
    rtsp_urls = cfg.get("rtsp_urls", [])
    loaders = []
    if rtsp_urls:
        total_cam = len(rtsp_urls)
        for i, url in enumerate(rtsp_urls):
            loaders.append(RTSPStreamLoader(url, f"cam_{i}").start())
    else:
        videos = np.array(os.listdir(cfg["video_path"]))
        total_cam = len(videos)
        for i in range(total_cam):
            cam[f"cam_{i}"] = cv2.VideoCapture(os.path.join(cfg["video_path"], videos[i]))
            cam[f"cam_{i}"].set(3, cfg["size_each_camera_image"][0])
            cam[f"cam_{i}"].set(4, cfg["size_each_camera_image"][1])
    
    if cfg["save_video_camera_tracking"]:
        out = cv2.VideoWriter(
            os.path.join(
                cfg["output_path_name_save_video_camera_tracking"],
                f'{cfg["output_name_save_video_camera_tracking"]}.avi',
            ),
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            cfg["fps_save_video_camera_tracking"],
            setup_resolution(
                cfg["size_each_camera_image"], cfg["resize_all_camera_image"], total_cam
            ),
        )

    frame_count = 0
    detected_ids_in_frame = []  # Track IDs detected in current frame
    
    # for _ in tqdm(while_true(), desc="Tracking person in progress..."):
    for _ in while_true():
        frame_count += 1
        current_time = datetime.now()  # Lấy timestamp cho frame hiện tại
        detected_ids_in_frame = []  # Reset cho frame mới
        
        # Set up variable
        images = {}
        predicts = {}

        # Get camera image
        if rtsp_urls:
            for i, loader in enumerate(loaders):
                frame, _ = loader.read()
                if frame is None:
                    continue
                images[f"image_{i}"] = frame
        else:
            for i in range(total_cam):
                ret, images[f"image_{i}"] = cam[f"cam_{i}"].read()
                if not ret:
                    print("\n Video đã kết thúc")
                    break

        if not images:
            if rtsp_urls:
                time.sleep(0.01)
                continue
            break

        # Predict person with object detection
        for i in range(total_cam):
            image_key = f"image_{i}"
            if image_key not in images:
                continue
            predicts[image_key] = object_detection.predict_img(images[image_key])

        # Resize image for display in screen (keep aspect ratio)
        display_images = {}
        resize_meta = {}
        for i in range(total_cam):
            # display_images[f"image_{i}"], scale, pad = resize_with_letterbox(
            #     images[f"image_{i}"],
            image_key = f"image_{i}"
            if image_key not in images:
                placeholder = np.zeros(
                    (
                        cfg["size_each_camera_image"][1],
                        cfg["size_each_camera_image"][0],
                        3,
                    ),
                    dtype=np.uint8,
                )
                display_images[image_key] = placeholder
                resize_meta[image_key] = (1.0, (0, 0))
                continue
            display_images[image_key], scale, pad = resize_with_letterbox(
                images[image_key],
                cfg["size_each_camera_image"],
            )
            #resize_meta[f"image_{i}"] = (scale, pad)
            resize_meta[image_key] = (scale, pad)

        for i in range(total_cam):
            image_key = f"image_{i}"
            if image_key not in images:
                continue
            if image_key not in predicts:
                continue
            for predict in predicts[image_key]:
                cls_name = tuple(predict.keys())[0]
                x1, y1, x2, y2 = predict[cls_name]["bounding_box"]

                # Person identification
                height, width = images[f"image_{i}"].shape[:2]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                cropped_image = images[f"image_{i}"][y1:y2, x1:x2]
                extracted_features = feature_extraction.predict_img(cropped_image)[0]

                # Add new person if data is empty
                if not detected_persons:
                    # ==================== LIFECYCLE: CREATE NEW PERSON ====================
                    match_info = {
                        'match_score': None,
                        'matched_global_id': None,
                        'match_confidence': None,
                        'reasoning': 'first_detection',
                        'feasibility_reason': 'no_existing_persons'
                    }
                    
                    person_id = lifecycle_manager.create_person(
                        camera_id=i,
                        confidence=predict[cls_name]["confidence"],
                        bbox=(x1, y1, x2, y2),
                        match_info=match_info
                    )
                    # ======================================================================
                    
                    detected_persons[f"id_{person_id}"] = {
                        "extracted_features": extracted_features,
                        "id": person_id,
                        "camera_id": i,
                        "cls_name": cls_name,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": predict[cls_name]["confidence"],
                        "color": np.random.randint(0, 255, size=3),
                    }
                    detected_ids_in_frame.append(person_id)
                else:
                    # ==================== TIME WINDOW FILTERING ====================
                    # Lấy các persons có thể match (trong time window)
                    matchable_persons = lifecycle_manager.get_matchable_persons(
                        current_time, 
                        time_window_seconds
                    )
                    
                    # Nếu không có person nào trong time window -> tạo mới
                    if not matchable_persons:
                        # Tất cả persons đều ngoài time window
                        if detected_persons:
                            lifecycle_manager.time_window_rejections += 1
                            print(f"    Time window: All persons outside window, creating new ID")
                        
                        person_id = lifecycle_manager.create_person(
                            camera_id=i,
                            confidence=predict[cls_name]["confidence"],
                            bbox=(x1, y1, x2, y2)
                        )
                        
                        detected_persons[f"id_{person_id}"] = {
                            "extracted_features": extracted_features,
                            "id": person_id,
                            "camera_id": i,
                            "cls_name": cls_name,
                            "bbox": (x1, y1, x2, y2),
                            "confidence": predict[cls_name]["confidence"],
                            "color": np.random.randint(0, 255, size=3),
                        }
                        detected_ids_in_frame.append(person_id)
                        continue
                    # ================================================================
                    
                    # ==================== SIMILARITY MATCHING (CHỈ TRONG TIME WINDOW) ====================
                    # Chỉ tính similarity với những persons trong time window
                    candidates = []
                    for person_id, person in matchable_persons.items():
                        if f"id_{person_id}" not in detected_persons:
                            continue
                        
                        value = detected_persons[f"id_{person_id}"]
                        
                        # Tính cosine similarity
                        score = distance.cosine(
                            np.mean(value["extracted_features"], axis=0)
                            if len(value["extracted_features"]) > 1
                            else value["extracted_features"].flatten(),
                            extracted_features.flatten(),
                        )
                        
                        candidates.append({
                            "id": value["id"],
                            "cls_name": value["cls_name"],
                            "color": value["color"],
                            "score": score,
                            "time_diff": person.get_time_since_last_seen(current_time)
                        })
                    
                    # Sắp xếp theo similarity score (thấp nhất = tốt nhất)
                    if candidates:
                        top1_person = sorted(candidates, key=lambda d: d["score"])[0]
                        
                        # Existing person detected (trong time window + similarity OK)
                        if top1_person["score"] < cfg["feature_extraction_threshold"]:
                            # ==================== LIFECYCLE: UPDATE EXISTING PERSON ====================
                            lifecycle_manager.update_person(
                                person_id=top1_person["id"],
                                camera_id=i,
                                confidence=predict[cls_name]["confidence"],
                                bbox=(x1, y1, x2, y2)
                            )
                            # ===========================================================================
                            
                            detected_persons[f"id_{top1_person['id']}"] = {
                                "extracted_features": np.vstack(
                                    (
                                        detected_persons[f"id_{top1_person['id']}"][
                                            "extracted_features"
                                        ],
                                        extracted_features,
                                    )
                                )
                                if detected_persons[f"id_{top1_person['id']}"][
                                    "extracted_features"
                                ].shape[0]
                                < cfg["max_gallery_set_each_person"]
                                else np.vstack(
                                    (
                                        extracted_features,
                                        detected_persons[f"id_{top1_person['id']}"][
                                            "extracted_features"
                                        ][1:],
                                    )
                                ),
                                "id": top1_person["id"],
                                "camera_id": i,
                                "cls_name": top1_person["cls_name"],
                                "bbox": (x1, y1, x2, y2),
                                "confidence": predict[cls_name]["confidence"],
                                "color": top1_person["color"],
                            }
                            detected_ids_in_frame.append(top1_person["id"])
                        else:
                            # Similarity score không đủ -> tạo person mới
                            person_id = lifecycle_manager.create_person(
                                camera_id=i,
                                confidence=predict[cls_name]["confidence"],
                                bbox=(x1, y1, x2, y2)
                            )
                            
                            detected_persons[f"id_{person_id}"] = {
                                "extracted_features": extracted_features,
                                "id": person_id,
                                "camera_id": i,
                                "cls_name": cls_name,
                                "bbox": (x1, y1, x2, y2),
                                "confidence": predict[cls_name]["confidence"],
                                "color": np.random.randint(0, 255, size=3),
                            }
                            detected_ids_in_frame.append(person_id)
                    else:
                        # Không có candidates hợp lệ -> tạo mới
                        person_id = lifecycle_manager.create_person(
                            camera_id=i,
                            confidence=predict[cls_name]["confidence"],
                            bbox=(x1, y1, x2, y2)
                        )
                        
                        detected_persons[f"id_{person_id}"] = {
                            "extracted_features": extracted_features,
                            "id": person_id,
                            "camera_id": i,
                            "cls_name": cls_name,
                            "bbox": (x1, y1, x2, y2),
                            "confidence": predict[cls_name]["confidence"],
                            "color": np.random.randint(0, 255, size=3),
                        }
                        detected_ids_in_frame.append(person_id)
                    # ===================================================================================

            # Draw all bbox - Chỉ vẽ active persons
            active_persons = lifecycle_manager.active_persons
            for person_id, person in active_persons.items():
                if person.current_camera == i and f"id_{person_id}" in detected_persons:
                    value = detected_persons[f"id_{person_id}"]
                    
                    # Chọn màu theo state
                    if person.state == PersonState.DETECTED:
                        color = (0, 255, 0)  # Xanh lá - mới
                    elif person.state == PersonState.TRACKING:
                        color = value["color"].tolist()  # Màu riêng
                    else:
                        color = (128, 128, 128)  # Xám
                    
                    scale, pad = resize_meta[f"image_{i}"]
                    disp_bbox = map_bbox_letterbox(
                        value["bbox"],
                        scale,
                        pad,
                        cfg["size_each_camera_image"],
                    )
                    cv2.rectangle(
                        display_images[f"image_{i}"],
                        disp_bbox[:2],
                        disp_bbox[2:],
                        color,
                        2,
                    )
                    
                    # Thêm state vào label
                    label = f"{value['cls_name']} {person_id} [{person.state.value[:4]}]: {value['confidence']:.2f}"
                    cv2.putText(
                        display_images[f"image_{i}"],
                        label,
                        (disp_bbox[0], max(0, disp_bbox[1] - 10)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        color,
                        2,
                    )

        # ==================== LIFECYCLE: PROCESS FRAME END ====================
        lifecycle_manager.process_frame_end(detected_ids_in_frame)
        # ======================================================================

        if not display_images:
            if rtsp_urls:
                time.sleep(0.01)
                continue
            break

        # Display all cam
        if total_cam % 2 == 0:
            display_image = stack_images(
                cfg["resize_all_camera_image"],
                (
                    [display_images[f"image_{i}"] for i in range(0, total_cam // 2)],
                    [
                        display_images[f"image_{i}"]
                        for i in range(total_cam // 2, total_cam)
                    ],
                ),
            )
        else:
            display_image = stack_images(
                cfg["resize_all_camera_image"],
                ([display_images[f"image_{i}"] for i in range(total_cam)],),
            )

        if cfg["save_video_camera_tracking"]:
            out.write(display_image)
        if cfg["display_video_camera_tracking"]:
            cv2.imshow("CCTV Misale", display_image)
            if cv2.waitKey(1) == ord("q"):
                print("\n Người dùng dừng chương trình")
                break
        
        # Print status mỗi 100 frames
        if frame_count % 100 == 0:
            lifecycle_manager.print_status()

    # ==================== KẾT THÚC - FINAL REPORT ====================
    print("\n" + "="*80)
    print("SAVING LIFECYCLE DATA...")
    print("="*80)
    
    # Print final report
    lifecycle_manager.print_final_report()
    
    # Save summary
    lifecycle_manager.save_summary()
    
    # Thống kê cuối
    stats = lifecycle_manager.get_statistics()
    print(f"\n Session Statistics:")
    print(f"   - Total persons seen: {stats['total_persons']}")
    print(f"   - Session duration: {stats['session_duration']:.1f} seconds")
    print(f"   - Active: {stats['active_persons']} | Lost: {stats['lost_persons']} | Archived: {stats['archived_persons']}")
    print(f"   - Time window rejections: {stats['time_window_rejections']}")
    print(f"   - Topology rejections: {stats['topology_rejections']}")
    print(f"   - Same camera matches: {stats['same_camera_matches']}")
    print(f"   - Topology transitions: {stats['topology_transitions']}")
    # =================================================================

    # Release all cam
    for loader in loaders:
        loader.stop()
    for i in range(total_cam):
        if f"cam_{i}" in cam:
            cam[f"cam_{i}"].release()
    if cfg["save_video_camera_tracking"]:
        out.release()
    cv2.destroyAllWindows()
    
    print("\n Hoàn tất!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source-config-file",
        default="./config.yaml",
        help="Input your config.yaml file",
    )
    value_parser = parser.parse_args()

    with open(value_parser.source_config_file, "r") as f:
        file_config = yaml.safe_load(f)
    main(file_config)
