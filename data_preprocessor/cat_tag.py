# coding: utf-8
from datetime import datetime
import time
import os
from bisect import bisect_left


class Label(object):
    def __init__(self):
        self.label_file = None

    def parse_label_file(self, path_to_label_file):
        self.label_file = dict()
        with open(path_to_label_file) as f:
            for l in f.readlines():
                anchor_id, user_id, labels, addtime = l.strip().split("\t")
                addtime = datetime.strptime(addtime[:-2], "%Y-%m-%d %H:%M:%S")
                ymd = datetime.strftime(addtime, "%Y%m%d")
                unixtime = int(time.mktime(addtime.timetuple()))
                if self.label_file.get(ymd) is None:
                    self.label_file[ymd] = list()
                self.label_file[ymd].append((user_id, anchor_id, labels, unixtime))

    def parse_qchat_watch_file(self, path_to_watch_file):
        watch_file = dict()
        with open(path_to_watch_file) as f:
            for l in f.readlines():
                user_id, anchor_id, live_id, watching_duration, endtime = l.strip().split("\t")
                if watch_file.get(":".join([user_id, anchor_id])) is None:
                    watch_file[":".join([user_id, anchor_id])] = list()
                watch_file[":".join([user_id, anchor_id])].append((int(endtime), int(watching_duration), l))
        return watch_file

    def attatch_tagging_to_watch(self, label, l):
        return '\t'.join([l, label])

    def match_label_file_with_watch_record(self, watch_file_dir, time_gap_threshold, dest_file):
        f = open(dest_file, 'w')
        for ymd in self.label_file:
            watch_file = self.parse_qchat_watch_file(os.path.join(watch_file_dir, ymd))
            for user_id, anchor_id, labels, tagging_unixtime in self.label_file[ymd]:
                watch_file_list = watch_file.get(":".join([user_id, anchor_id]))
                if watch_file_list is None:
                    continue
                qchat_endtime_list = [x[0] for x in watch_file_list]
                qchat_watchtime_list = [x[1] for x in watch_file_list]
                if len(qchat_endtime_list) > 0:
                    i = bisect_left(qchat_endtime_list, tagging_unixtime)
                    if i != len(qchat_endtime_list):
                        if tagging_unixtime - qchat_endtime_list[i] < time_gap_threshold and time_gap_threshold > qchat_watchtime_list[i]:
                            f.write(self.attatch_tagging_to_watch(labels, watch_file_list[i][2]))
                            f.write("\n")
                        if tagging_unixtime > qchat_watchtime_list[i + 1] - time_gap_threshold - qchat_watchtime_list[i + 1]:
                            f.write(self.attatch_tagging_to_watch(labels, watch_file_list[i + 1][2]))
                            f.write("\n")
        f.close()


if __name__ == "__main__":
    label_gen = Label()
    label_gen.parse_label_file()