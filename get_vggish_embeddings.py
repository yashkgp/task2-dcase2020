import sys
import os

if __name__ == '__main__':
	# path1 = machine type
	# path2 = train/test
	path1 = sys.argv[1]
	path2 = sys.argv[2]
	os.chdir('vggish')
	cmd = os.path.join('python vggish_inference_demo.py --wav_file ../data',  path1, path2, ' --tfrecord_file ../data', path1, path1+'_'+path2+'_vggish_feats')
	os.system(cmd)
