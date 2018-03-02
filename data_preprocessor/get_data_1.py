# coding: utf-8
import os
import logging


def open_list(filename):
	file_set = set()
	with open(filename) as f:
		for l in f.readlines():
			file_set.add(l.strip())
	return file_set


def read_watch_file(dirname, user_watch_time, user_list, anchor_list):
	logging.info("start loading {}".format(dirname))
	for filename in os.listdir(dirname):
		if filename != "_SUCCESS":
			logging.info("start reading file {}".format(os.path.join(dirname, filename)))
			with open(os.path.join(dirname, filename)) as f:
				for line in f.readlines():
					tmp = line.strip().split('\t')
					user = tmp[0]
					anchor = tmp[1]
					watch_time = float(tmp[-1])
					if user in user_list and anchor in anchor_list:
						if user_watch_time.get(user) is None:
							user_watch_time[user] = dict()
						if user_watch_time[user].get(anchor) is None:
							user_watch_time[user][anchor] = [watch_time]
						else:
							user_watch_time[user][anchor].append(watch_time)
			logging.info("File {} loaded".format(os.path.join(dirname, filename)))
	logging.info("done.")

def read_watch_file_list(dirname_list, user_list_file, anchor_list_file):
	logging.info("loading user and anchor list.")
	user_list = open_list(user_list_file)
	anchor_list = open_list(anchor_list_file)
	logging.info("done.")
	
	user_watch_time = dict()
	for dirname in dirname_list:
		read_watch_file(dirname, user_watch_time, user_list, anchor_list)

	for user in user_watch_time:
		for anchor in user_watch_time[user]:
			user_watch_time[user][anchor] = sum(user_watch_time[user][anchor]) / len(user_watch_time[user][anchor])

	return user_watch_time

def save_user_watch_time(train_file, user_watch_time):
	with open(train_file, 'w') as f:
		for user in user_watch_time:
			for anchor in user_watch_time[user]:
				f.write(user + "\t" + anchor + "\t" + str(int(user_watch_time[user][anchor])) + "\n")

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s:%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %A %H:%M:%S')
	dirname_list = [os.path.join("E:\\liushen_watching_time_md5", str(x)) for x in range(20171220, 20171228)]
	user_list_file = "E:\\user_list"
	anchor_list_file = "E:\\anchor_list"

	user_watch_time = read_watch_file_list(dirname_list, user_list_file, anchor_list_file)
	save_user_watch_time("E:\\train_data", user_watch_time)