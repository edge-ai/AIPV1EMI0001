from typing import Any, List
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
        self.max_frames_in_batch = 0
        self.num_frames_in_batch = 0
        self.objInROI_labels = {}
        self.objLCCurrCnt_labels = {}
        
    def parse_batch_meta(self, data_dict: Any) -> Any:
        self.max_frames_in_batch = data_dict['batch_meta']['max_frames_in_batch']
        self.num_frames_in_batch = data_dict['batch_meta']['num_frames_in_batch']
        if self.num_frames_in_batch < self.max_frames_in_batch:
            mlogger.info(f"Missing frames: {self.num_frames_in_batch} / {self.max_frames_in_batch}")

        frame_data = []
        for i in range(self.max_frames_in_batch):
            frame = {}
            frame['source_id'] = i
            frame['frame_num'] = None
            frame['num_obj_meta'] = None
            frame['ntp_timestamp'] = None
            if self.pgie_classes is not None:
                frame['obj_counter'] = [None]*len(self.pgie_classes)
            else:
                frame['obj_counter'] = []
            frame['objInROIcnt'] = None
            frame['objLCCurrCnt'] = None
            frame_data.append(frame)
        
        for frame_meta in data_dict['batch_meta']['frame_meta_list']:
            frame = {}
            frame['source_id'] = frame_meta['source_id']
            frame['frame_num'] = frame_meta['frame_num']
            frame['num_obj_meta'] = frame_meta['num_obj_meta']
            frame['ntp_timestamp'] = frame_meta['ntp_timestamp']

            if self.pgie_classes is not None:
                obj_counter = [0]*len(self.pgie_classes)
                for obj_meta in frame_meta['obj_meta_list']:
                    obj_counter[obj_meta['class_id']] += 1
                frame['obj_counter'] = obj_counter
            else:
                frame['obj_counter'] = []

            if len(frame_meta['frame_user_meta_list']):
                frame['objInROIcnt'] = frame_meta['frame_user_meta_list'][0]['objInROIcnt']
                frame['objLCCurrCnt'] = frame_meta['frame_user_meta_list'][0]['objLCCurrCnt']      
            else:
                frame['objInROIcnt'] = {}
                frame['objLCCurrCnt'] = {}
            
            frame_data[frame['source_id']] = frame

        return frame_data
        
    def merge_and_analyze(self, frame_data_list: List[Any], frame_time_list: List[float]) -> Any:
        merged_data = {}
        
        frame_data_array = np.array(frame_data_list, dtype=object).T
        for source_frames in frame_data_array:
            if len(source_frames):
                source_id = source_frames[0]['source_id']
                merged_data[source_id] = {}
                merged_data[source_id]['source_id'] = source_frames[0]['source_id']
                merged_data[source_id]['frame_number_from'] = source_frames[0]['frame_num']
                merged_data[source_id]['frame_number_to'] = source_frames[-1]['frame_num']
                merged_data[source_id]['ntp_timestamp_from'] = source_frames[0]['ntp_timestamp']
                merged_data[source_id]['ntp_timestamp_to'] = source_frames[-1]['ntp_timestamp']

                obj_counter_list = []
                objInROIcnt_list = []
                objLCCurrCnt_list = []
                for frame in source_frames:
                    obj_counter_list.append(frame['obj_counter'])
                    objInROIcnt_list.append(frame['objInROIcnt'])
                    objLCCurrCnt_list.append(frame['objLCCurrCnt'])
                    
                obj_counter_array = np.array(obj_counter_list, dtype=float).T
                max_array = np.nanmax(obj_counter_array, axis=1)
                min_array = np.nanmin(obj_counter_array, axis=1)
                mean_array = np.nanmean(obj_counter_array, axis=1)
                median_array = np.nanmedian(obj_counter_array, axis=1)
                sum_array = np.nansum(obj_counter_array, axis=1)
                summary = np.around(np.vstack((max_array, min_array, mean_array, median_array, sum_array)).T).astype(int)
                
                
                obj_counter = {}
                for class_idx, class_summary in enumerate(summary):
                    if self.pgie_classes is not None:
                        class_name = self.pgie_classes[class_idx]
                    else:
                        class_name = str(class_idx)
                    obj_counter[class_name] = {
                        'max_val': int(class_summary[0]),
                        'min_val': int(class_summary[1]),
                        'mean_val': int(class_summary[2]),
                        'median_val': int(class_summary[3]),
                        'count_val': int(class_summary[4]),
                    }
                merged_data[source_id]['obj_counter'] = obj_counter

                keys = []
                if not source_id in self.objInROI_labels.keys():
                    self.objInROI_labels[source_id] = []
                for dict in objInROIcnt_list:
                    if dict is not None:
                        keys = list(dict.keys())
                        for key in keys:
                            if key not in self.objInROI_labels[source_id]:
                                self.objInROI_labels[source_id].append(key)
                objInROI_summary = {}
                for label in self.objInROI_labels[source_id]:
                    objInROI_summary[label] = {'max_val': 0, 'min_val': 0, 'sum_val': 0}
                    if label in keys:
                        values_array = np.array([dict[label] for dict in objInROIcnt_list if dict is not None])
                        if len(values_array) > 0:
                            max_val = int(np.max(values_array))
                            min_val = int(np.min(values_array))
                            sum_val = int(np.sum(values_array))
                            objInROI_summary[label] = {'max_val': max_val, 'min_val': min_val, 'sum_val': sum_val}

                keys = []
                if not source_id in self.objLCCurrCnt_labels.keys():
                    self.objLCCurrCnt_labels[source_id] = []
                for dict in objLCCurrCnt_list:
                    if dict is not None:
                        keys = list(dict.keys())
                        for key in keys:
                            if key not in self.objLCCurrCnt_labels[source_id]:
                                self.objLCCurrCnt_labels[source_id].append(key)
                objLCCurrCnt_summary = {}
                for label in self.objLCCurrCnt_labels[source_id]:
                    objLCCurrCnt_summary[label] = {'max_val': 0, 'min_val': 0, 'sum_val': 0}
                    if label in keys:
                        values_array = np.array([dict[label] for dict in objLCCurrCnt_list if dict is not None])
                        if len(values_array) > 0:
                            max_val = int(np.max(values_array))
                            min_val = int(np.min(values_array))
                            sum_val = int(np.sum(values_array))
                            objLCCurrCnt_summary[label] = {'max_val': max_val, 'min_val': min_val, 'sum_val': sum_val}
                    
                merged_data[source_id]['analytics_obj_cnt'] = {'objInROIcnt': objInROI_summary, 'objLCCurrCnt': objLCCurrCnt_summary}
                
        return merged_data