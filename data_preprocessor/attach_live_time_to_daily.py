# coding: utf-8
import os
from datetime import datetime, timedelta
import time

live_info_dir = "E:\\huajiao_data\\live"


class Attach(object):
    def __init__(self):
        self._live_dict = None
        self._live_dict_day = None
        self.label_file = None

    def parse_label_file(self, path_to_label_file):
        self.label_file = dict()
        with open(path_to_label_file, encoding="utf-8") as f:
            for l in f.readlines():
                _, anchor_id, user_id, labels, addtime, _ = l.strip().split("\t")
                addtime = datetime.strptime(addtime[:-2], "%Y-%m-%d %H:%M:%S")
                ymd = datetime.strftime(addtime, "%Y%m%d")
                unixtime = int(time.mktime(addtime.timetuple()))
                if self.label_file.get(":".join([user_id, anchor_id])) is None:
                    self.label_file[":".join([user_id, anchor_id])] = list()
                self.label_file[":".join([user_id, anchor_id])].append((labels, unixtime))

    def live_dict(self, day):
        if self._live_dict is None or self._live_dict_day != day:
            self._live_dict = dict()
            with open(os.path.join(live_info_dir, "live_hot_" + day + ".csv"), encoding='utf-8') as f:
                for l in f.readlines():
                    tmp = l.strip().split('\t')
                    liveid = tmp[1]
                    live_addtime = tmp[-5]
                    live_endtime = tmp[-4]
                    if live_endtime == "0000-00-00 00:00:00":
                        live_endtime = datetime.strptime(live_addtime, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=float(tmp[3]))
                        live_endtime = live_endtime.strftime("%Y-%m-%d %H:%M:%S")
                    self._live_dict[liveid] = (live_addtime, live_endtime)
        self._live_dict_day = day
        return self._live_dict

    def attach_time(self, filename):
        outfile = open(filename + "_ext", "w")
        anchor_set = set()
        user_set = set()
        with open("D:\\Code\\huajiao\\raw_data\\anchor_list") as f:
            for l in f.readlines():
                anchor_set.add(l.strip())
        with open("D:\\Code\\huajiao\\raw_data\\user_list") as f:
            for l in f.readlines():
                user_set.add(l.strip())
        with open(filename) as f:
            for l in f.readlines():
                userid, anchorid, liveid, _, _, _, _, _, _, day = l.strip().split('\t')
                if anchorid not in anchor_set or userid not in user_set:
                    continue
                live_dict = self.live_dict(day)
                add_time, end_time = live_dict.get(liveid, [0, 0])
                if add_time == 0:
                    continue
                outfile.write(l.strip() + "\t" + add_time + "\t" + end_time + "\n")
        outfile.close()

    def attach_labels(self, labelfile, filename):
        self.parse_label_file(labelfile)
        outfile = open(filename + "_label", "w")
        with open(filename) as f:
            for l in f.readlines():
                userid, anchorid, liveid, _, _, _, _, _, _, day, add_time, end_time = l.strip().split('\t')
                label_list = self.label_file.get(":".join([userid, anchorid]))
                if label_list is not None:
                    for labels, unixtime in label_list:
                        addtime = int(time.mktime(datetime.strptime(add_time, "%Y-%m-%d %H:%M:%S").timetuple()))
                        endtime = int(time.mktime(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timetuple()))
                        if (unixtime >= addtime) and (unixtime <= endtime):
                            outfile.write(l.strip() + '\t' + labels + "\n")
                            l = l.strip() + "\t" + labels + "\n"
                outfile.write(l)
        outfile.close()


if __name__ == "__main__":
    at = Attach()
    # at.attach_time("D:\\Code\\huajiao\\raw_data\\train_data_label")
    at.attach_labels("D:\\Code\\huajiao\\raw_data\\tagging_watch_train",
                     "D:\\Code\\huajiao\\raw_data\\train_data_label_ext")

    # at.attach_time("D:\\Code\\huajiao\\raw_data\\test_data_label")
    at.attach_labels("D:\\Code\\huajiao\\raw_data\\tagging_watch_test",
                     "D:\\Code\\huajiao\\raw_data\\test_data_label_ext")
