from typing import Any, List
from numbers import Integral

import numpy as np
from aiprocessors.callback.custom import CustomCallback
from common.logmanager import get_logger

mlogger = get_logger(name=__name__)


class UserCustomCallback(CustomCallback):
    def initialize(self):
        self.data_parser_config = {
            "frame_meta_list":          True,    # For PGIE
            "obj_meta_list":            True,    # For PGIE
            "classifier_meta_list":     False,   # For SGIE
            "label_info_list":          False,   # For SGIE
            "frame_user_meta_list":     True,    # For NvDsAnalytics
            "obj_user_meta_list":       True,    # For NvDsAnalytics
            "display_meta_list":        False,   # For NvOSD
        }

    def parse_batch_meta(self, data_dict: Any) -> Any:
        try:
            batch = data_dict['batch_meta']
            max_frames = batch['max_frames_in_batch']
            num_frames = batch['num_frames_in_batch']
            frame_meta_list = batch['frame_meta_list']
        except (KeyError, TypeError) as error:
            mlogger.warning(f"Invalid batch metadata: {error}")
            return []

        if (not isinstance(max_frames, Integral) or isinstance(max_frames, bool)
                or max_frames < 0 or not isinstance(num_frames, Integral)
                or isinstance(num_frames, bool) or num_frames < 0
                or not isinstance(frame_meta_list, list)):
            mlogger.warning("Invalid batch metadata: expected non-negative frame counts and a frame list")
            return []
        classes = self.pgie_classes or []
        configured_sources = getattr(self, 'input_sources', None)
        if configured_sources and not isinstance(configured_sources, dict):
            mlogger.warning("Invalid configured input sources")
            return []
        if configured_sources:
            if any(not isinstance(source_id, Integral) or isinstance(source_id, bool) or source_id < 0
                   for source_id in configured_sources):
                mlogger.warning("Invalid configured source_id")
                return []
            source_ids = sorted(int(source_id) for source_id in configured_sources)
        else:
            source_ids = range(max_frames)
        if num_frames < len(source_ids):
            mlogger.info(f"Missing frames: {num_frames} / {len(source_ids)}")

        frame_data = {
            source_id: {
                'source_id': source_id,
                'available': False,
                'frame_num': None,
                'num_obj_meta': None,
                'ntp_timestamp': None,
                'obj_counter': [None] * len(classes),
                'objInROIcnt': None,
                'objLCCurrCnt': None,
            }
            for source_id in source_ids
        }

        for frame_meta in frame_meta_list:
            if not isinstance(frame_meta, dict):
                mlogger.warning("Skipping invalid frame metadata")
                continue
            source_id = frame_meta.get('source_id')
            if (not isinstance(source_id, Integral) or isinstance(source_id, bool)
                    or source_id not in frame_data):
                mlogger.warning(f"Skipping frame with invalid source_id: {source_id!r}")
                continue
            obj_meta_list = frame_meta.get('obj_meta_list')
            if frame_meta.get('frame_num') is None or not isinstance(obj_meta_list, list):
                mlogger.warning(f"Skipping incomplete frame metadata for source {source_id}")
                continue

            obj_counter = [0] * len(classes)
            for obj_meta in obj_meta_list:
                class_id = obj_meta.get('class_id') if isinstance(obj_meta, dict) else None
                if (isinstance(class_id, Integral) and not isinstance(class_id, bool)
                        and 0 <= class_id < len(classes)):
                    obj_counter[class_id] += 1
                else:
                    mlogger.warning(f"Ignoring invalid class_id {class_id!r} for source {source_id}")

            analytics = {'objInROIcnt': {}, 'objLCCurrCnt': {}}
            user_meta_list = frame_meta.get('frame_user_meta_list', [])
            if not isinstance(user_meta_list, list):
                mlogger.warning(f"Invalid analytics metadata for source {source_id}")
                user_meta_list = []
            for user_meta in user_meta_list:
                if not isinstance(user_meta, dict):
                    mlogger.warning(f"Invalid analytics metadata for source {source_id}")
                    continue
                for key in analytics:
                    values = user_meta.get(key)
                    if isinstance(values, dict):
                        analytics[key].update(values)
                    elif values is not None:
                        mlogger.warning(f"Invalid {key} metadata for source {source_id}")

            frame_data[source_id] = {
                'source_id': int(source_id),
                'available': True,
                'frame_num': frame_meta['frame_num'],
                'num_obj_meta': frame_meta.get('num_obj_meta'),
                'ntp_timestamp': frame_meta.get('ntp_timestamp'),
                'obj_counter': obj_counter,
                **analytics,
            }

        return list(frame_data.values())

    def merge_and_analyze(self, frame_data_list: List[Any], frame_time_list: List[float]) -> Any:
        sources = {}
        for batch in frame_data_list:
            for frame in batch:
                sources.setdefault(frame['source_id'], []).append(frame)

        classes = self.pgie_classes or []
        stat_names = ('max_val', 'min_val', 'mean_val', 'median_val', 'count_val')
        merged_data = {}
        for source_id, source_frames in sources.items():
            observed = [frame for frame in source_frames if frame['available']]
            if observed and classes:
                counts = np.asarray([frame['obj_counter'] for frame in observed], dtype=float)
                summary = np.rint(np.stack((
                    np.max(counts, axis=0),
                    np.min(counts, axis=0),
                    np.mean(counts, axis=0),
                    np.median(counts, axis=0),
                    np.sum(counts, axis=0),
                ), axis=1)).astype(int)
                obj_counter = {
                    label: {name: int(value) for name, value in zip(stat_names, summary[index])}
                    for index, label in enumerate(classes)
                }
            else:
                obj_counter = {
                    label: {name: None for name in stat_names}
                    for label in classes
                }

            analytics = {}
            for key in ('objInROIcnt', 'objLCCurrCnt'):
                values_by_label = {}
                for frame in observed:
                    for label, value in frame[key].items():
                        if isinstance(value, Integral) and not isinstance(value, bool):
                            values_by_label.setdefault(label, []).append(int(value))
                        else:
                            mlogger.warning(f"Ignoring invalid {key} value for source {source_id}, label {label!r}")
                analytics[key] = {
                    label: {'max_val': max(values), 'min_val': min(values), 'sum_val': sum(values)}
                    for label, values in values_by_label.items()
                }

            first = observed[0] if observed else {}
            last = observed[-1] if observed else {}
            merged_data[source_id] = {
                'source_id': source_id,
                'available': bool(observed),
                'frame_number_from': first.get('frame_num'),
                'frame_number_to': last.get('frame_num'),
                'ntp_timestamp_from': first.get('ntp_timestamp'),
                'ntp_timestamp_to': last.get('ntp_timestamp'),
                'obj_counter': obj_counter,
                'analytics_obj_cnt': analytics,
            }

        return merged_data