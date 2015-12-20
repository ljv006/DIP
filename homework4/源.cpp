#include <cv.h>
#include <highgui.h>
#include<math.h>
#include<time.h>
#include<iostream>
#include<algorithm>
using namespace std;
using namespace cv;
#define PI 3.1415926

//������ֵ�˲���
double* ave1 = new double[10], *ave2 = new double[82];
//�˲�����ʼ��
void init() {
	for (int i = 0; i < 9; i++) {
		ave1[i] = 1.0 / 9.0;
	}
	for (int i = 0; i < 81; i++) {
		ave2[i] = 1.0 / 81.0;
	}

}

/**�������ܣ���RGBͨ���ֱ����ֱ��ͼ���⻯���ٺϲ�
   ���룺input_img ����ͼ��
   ������������ͨ����ͼ��
*/
Mat equalize_hist1(Mat input_img) {
	//��ȡĿ���С�Ŀ�͸�
	int h = input_img.rows, w = input_img.cols;
	//�½�������ͨ����8λ���ͼƬ,һ��Ϊԭͼת�ɵ�ͨ�����ͼ����һ��Ϊ�����Ľ��
	Mat img1(h, w, CV_8UC1, Scalar(0, 0, 0)), res(h, w, CV_8UC1, Scalar(0, 0, 0));
	//��ԭ������ͨ����8λ���ͼƬת��Ϊ��ͨ����
	img1 = input_img;
	//����ֱ��ͼ���⻯
	int NumPixel[256]; //ͳ�Ƹ��Ҷ���Ŀ����256���Ҷȼ�
	for (int i = 0; i < 256; i++) {
		NumPixel[i] = 0;
	}
	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			Scalar intensity = img1.at<uchar>(i, j);
			int tmp = intensity.val[0];
			NumPixel[tmp] += 1; //��Ӧ�Ҷ�ֵ���ص���������һ
		}
	}
	//����Ҷȷֲ��ܶ�
	double ProbPixel[256];
	for (int i = 0; i < 256; i++) {
		ProbPixel[i] = 0;
	}
	for (int i = 0; i < 256; i++) {
		ProbPixel[i] = NumPixel[i] / (img1.cols * img1.rows * 1.0);
	}
	//�����ۼ�ֱ��ͼ�ֲ�
	double CumuPixel[256];
	for (int i = 0; i < 256; i++) {
		CumuPixel[i] = 0;
	}
	for (int i = 0; i < 256; i++) {
		if (i == 0) {
			CumuPixel[i] = ProbPixel[i];
		}
		else {
			CumuPixel[i] = CumuPixel[i - 1] + ProbPixel[i];
		}
	}
	//�ۼƷֲ�ȡ��
	for (int i = 0; i < 256; i++) {
		CumuPixel[i] = 255 * CumuPixel[i] + 0.5;
	}

	//�ԻҶ�ֵ����ӳ�䣨���⻯��
	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			res.at<uchar>(i, j) = CumuPixel[img1.at<uchar>(i, j)];
		}
	}
	return res;
}
/**�������ܣ��ۺϿ���RGB��ֱ��ͼ�������ƽ��ֱ��ͼ�Լ�ת���������ٶ�ÿ��ͨ�����д���
   ���룺R��G��B ����ͨ����ͼ��
   ���������ϲ����ɫͼ��
*/
Mat equalize_hist2(Mat R, Mat G, Mat B) {
	//��ȡĿ���С�Ŀ�͸�
	int h = R.rows, w = R.cols;
	//�½�������ͨ����8λ���ͼƬ,һ��Ϊԭͼת�ɵ�ͨ�����ͼ����һ��Ϊ�����Ľ��
	Mat res(h, w, CV_8UC3, Scalar(0, 0, 0));
	//����ֱ��ͼ���⻯
	int NumPixelR[256], NumPixelG[256], NumPixelB[256]; //ͳ�Ƹ��Ҷ���Ŀ����256���Ҷȼ�
	for (int i = 0; i < 256; i++) {
		NumPixelR[i] = 0;
		NumPixelB[i] = 0;
		NumPixelG[i] = 0;
	}
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			Scalar intensity = R.at<uchar>(i, j);
			int tmp = intensity.val[0];
			NumPixelR[tmp] += 1; //��ɫͨ����Ӧ���ص���������һ
			intensity = G.at<uchar>(i, j);
			tmp = intensity.val[0];
			NumPixelG[tmp] += 1; //��ɫͨ����Ӧ���ص���������һ
			intensity = B.at<uchar>(i, j);
			tmp = intensity.val[0];
			NumPixelB[tmp] += 1; //��ɫͨ����Ӧ���ص���������һ
		}
	}
	//����Ҷȷֲ��ܶ�
	double ProbPixel[256];
	for (int i = 0; i < 256; i++) {
		ProbPixel[i] = 0;
	}
	for (int i = 0; i < 256; i++) {
		ProbPixel[i] = (NumPixelR[i] + NumPixelB[i] + NumPixelG[i]) /( 3 * h * w * 1.0);
	}
	//�����ۼ�ֱ��ͼ�ֲ�
	double CumuPixel[256];
	for (int i = 0; i < 256; i++) {
		CumuPixel[i] = 0;
	}
	for (int i = 0; i < 256; i++) {
		if (i == 0) {
			CumuPixel[i] = ProbPixel[i];
		}
		else {
			CumuPixel[i] = CumuPixel[i - 1] + ProbPixel[i];
		}
	}
	//�ۼƷֲ�ȡ��
	for (int i = 0; i < 256; i++) {
		CumuPixel[i] = 255 * CumuPixel[i] + 0.5;
	}
	//�ԻҶ�ֵ����ӳ�䣨���⻯��
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			R.at<uchar>(i, j) = CumuPixel[R.at<uchar>(i, j)];
			G.at<uchar>(i, j) = CumuPixel[G.at<uchar>(i, j)];
			B.at<uchar>(i, j) = CumuPixel[B.at<uchar>(i, j)];
		}
	}
	//�ϲ�����ͨ�����õ���ɫͼ��
	vector<Mat>v = { B, G, R };
	merge(v, res);
	return res;
}
//������ƽ����
//���ڱ�Ե���������˸ı���س��˲�����С�Ĳ���
double calculate(double* input_img,int h, int w, double* filter, int x, int y, int len) {
	double sum = 0;
	for (int i = 0; i < len; ++i){
		for (int j = 0; j < len; ++j){
			//�ж��Ƿ񳬳���ͼƬ�ı߽�
			if (((x - 1 + i) >= 0) && ((y - 1 + j) >= 0) && ((x - 1 + i) < h) && ((y - 1 + j) < w)){
				sum = sum + filter[len * i + j] * input_img[(x - 1 + i) * w + y - 1 + j];
			}
		}
	}
	return sum;
}
//�㼸��ƽ����
double calculate1(double* input_img, int h, int w, int x, int y, int len) {
	double acc = 1;
	for (int i = 0; i < len; ++i){
		for (int j = 0; j < len; ++j){
			//�ж��Ƿ񳬳���ͼƬ�ı߽�
			if (((x - 1 + i) >= 0) && ((y - 1 + j) >= 0) && ((x - 1 + i) < h) && ((y - 1 + j) < w)){
				acc = acc * input_img[(x - 1 + i) * w + y - 1 + j];
			}
		}
	}
	acc = pow(acc, 1.0 / (len * len * 1.0));
	return acc;
}
//����ֵ
double calculate2(double* input_img, int h, int w, int x, int y, int len) {
	vector<double> v;
	for (int i = 0; i < len; ++i){
		for (int j = 0; j < len; ++j){
			//�ж��Ƿ񳬳���ͼƬ�ı߽�
			if (((x - 1 + i) >= 0) && ((y - 1 + j) >= 0) && ((x - 1 + i) < h) && ((y - 1 + j) < w)){
				v.push_back(input_img[(x - 1 + i) * w + y - 1 + j]);
			}
		}
	}
	sort(v.begin(), v.end());
	//����ֵʱҪ������ż���
	if (v.size() % 2)
		return (v[(v.size() + 1) / 2] + v[(v.size() + 1) / 2 - 1]) / 2.0;
	else return v[(v.size() + 1) / 2 - 1];
}
//����Сֵ
double calculate3(double* input_img, int h, int w, int x, int y, int len) {
	vector<double> v;
	for (int i = 0; i < len; ++i){
		for (int j = 0; j < len; ++j){
			//�ж��Ƿ񳬳���ͼƬ�ı߽�
			if (((x - 1 + i) >= 0) && ((y - 1 + j) >= 0) && ((x - 1 + i) < h) && ((y - 1 + j) < w)){
				v.push_back(input_img[(x - 1 + i) * w + y - 1 + j]);
			}
		}
	}
	sort(v.begin(), v.end());
	return *v.begin();
}
//�����ֵ
double calculate4(double* input_img, int h, int w, int x, int y, int len) {
	vector<double> v;
	for (int i = 0; i < len; ++i){
		for (int j = 0; j < len; ++j){
			//�ж��Ƿ񳬳���ͼƬ�ı߽�
			if (((x - 1 + i) >= 0) && ((y - 1 + j) >= 0) && ((x - 1 + i) < h) && ((y - 1 + j) < w)){
				v.push_back(input_img[(x - 1 + i) * w + y - 1 + j]);
			}
		}
	}
	sort(v.begin(), v.end());
	return *v.rbegin();
}
/**�������ܣ���˫����������б궨����ת��ΪͼƬ��ʽ��
   ���룺input_img ˫��������,�洢���ص���Ϣ
         h ͼƬ�ĸ߶�
		 w ͼƬ�Ŀ��
   ������궨ת�����ͼƬ
*/
Mat scale(double* input_img, int h, int w) {
	Mat tmp(h, w, CV_8UC1, Scalar(0, 0, 0));
	double max = -1000000;
	double min = 1000000;
	double shift, scale;
	//�ҳ����ص�Ҷ�ֵ�����ֵ����Сֵ
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++){
			input_img[i * w + j] = abs(input_img[i * w + j]);
			if (max < input_img[i * w + j]) max = input_img[i * w + j];
			if (min > input_img[i * w + j]) min = input_img[i * w + j];
		}
	}
	//���б궨����
	scale = 255.0 / (max - min);
	shift = -min * scale;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			tmp.at<uchar>(i, j) = scale * input_img[i * w + j] + shift;
		}
	}
	return tmp;
}
/*�������ܣ������˲�������ͼ����д���
  ���룺input_img ˫�������飬�洢���ص����Ϣ
        h ͼƬ�߶�
		w ͼƬ���
		filter ˫�������飬�洢�˲�������Ϣ
		len �˲����ı߳�
*/
double* filter2d(double* input_img,int h, int w, double* filter, int len) {
	double *res = new double[w * h];
	for (int i = 0; i < h; ++i){
		for (int j = 0; j < w; ++j){
			res[i*w+j] = calculate(input_img, h, w, filter, i, j, len);
		}
	}
	return res;
}


/**�������ܣ�������ͼƬ�ϲ�����˹����
   ���룺input_img ����ͼƬ�ĻҶ�ֵ��Ϣ
         h  ͼƬ�ĸ߶�
		 w  ͼƬ�Ŀ��
		 mean  ��˹�����ľ�ֵ
		 stdvar ��˹�����ı�׼��
   ����������˸�˹������ͼƬ
*/
double* noiseGenerator1(double* input_img, int h, int w, double mean, double stdvar) {
	double *res = new double[w * h];
	srand((unsigned)time(NULL));
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			double ra1 = rand() % 1000 / 1000.0;
			double ra2 = rand() % 1000 / 1000.0;
			double result = sqrt((-2) * log(ra2)) * sin(2 * PI * ra1);
			result = mean + stdvar * result;
			if (input_img[i * w + j] + result > 255) res[i*w + j] = 255;
			else if (input_img[i * w + j] + result < 0) res[i*w + j] = 0;
			else res[i*w + j] = input_img[i * w + j] + result;
		}
	}
	return res;
}
/**�������ܣ�������ͼƬ�ϲ���������
   ���룺input_img ����ͼƬ�ĻҶ�ֵ��Ϣ
         h  ͼƬ�ĸ߶�
		 w  ͼƬ�Ŀ��
		 pro  �����������ĸ���
   �������������������ͼƬ
*/
double* noiseGenerator2(double* input_img, int h, int w, double pro) {
	double *res = new double[w * h];
	srand((unsigned)time(NULL));
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			double rd = rand() % 1000 / 1000.0;
			if (rd < pro) {
				res[i * w + j] = 255;
			}
			else {
				res[i * w + j] = input_img[i * w + j];
			}
		}
	}
	return res;
}
/**�������ܣ�������ͼƬ�ϲ�����������
   ���룺input_img ����ͼƬ�ĻҶ�ֵ��Ϣ
         h  ͼƬ�ĸ߶�
		 w  ͼƬ�Ŀ��
		 pro  �������������������ĸ���
   ����������˽���������ͼƬ
*/
double* noiseGenerator3(double* input_img, int h, int w, double pro) {
	double *res = new double[w * h];
	srand((unsigned)time(NULL));
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			double rd = rand() % 1000 / 1000.0;
			if (rd < pro) {
				res[i * w + j] = 255;
			}
			else if (rd < 2 * pro){
				res[i * w + j] = 0;
			}
			else {
				res[i * w + j] = input_img[i * w + j];
			}
		}
	}
	return res;
}



int main() {
	init();
	//�洢���������
	string cmd;
	//�����ļ���
	string file;
	while (true) {
		//����ͼƬ�ļ�
		cout << "Enter the image name\n";
		cin >> file;
		Mat input = imread("./" + file);
		//���������ͼƬ�ǲ�ɫͼƬ���ǻҶ�ͼ
		cout << "1---gray image\n"
			<< "2---color image\n";
		cin >> cmd;
		if (cmd == "1") {
			cvtColor(input, input, CV_RGB2GRAY);
		}
		int h = input.rows, w = input.cols;
		double *tmp = new double[h * w];
		double *tmp1 = new double[h * w];
		double *tmp2 = new double[h * w];
		double *res1 = new double[h * w];
		double *res2 = new double[h * w];
		double *tmp3 = new double[h * w];
		double *tmp4 = new double[h * w];
		double *r = new double[h * w];
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				Scalar a = input.at<uchar>(i, j);
				tmp[i * w + j] = a.val[0];
			}
		}
		imshow("input", input);
		waitKey(0);
		cout << "a---add noise\n"
			<< "f---filter\n"
			<< "h---histogram equalization\n";
		cin >> cmd;
		//�������
		if (cmd == "a") {
				cout << "1---add gaussian noise\n"
				<< "2---add salt noise\n"
				<< "3---add salt-and-pepper noise\n";
			cin >> cmd;
			//��Ӹ�˹����
			if (cmd == "1") {
				cout << "Enter the mean and standard deviation\n";
				double mean, deviation;
				cin >> mean >> deviation;
				Mat m1 = scale(noiseGenerator1(tmp, h, w, mean, deviation), h, w);
				imshow("Gaussian", m1);
				waitKey(0);
				cout << "Save the image with the file name:\n";
				string f;
				cin >> f;
				imwrite("./" + f, m1);
			}
			//���������
			if (cmd == "2") {
				cout << "Enter the probability\n";
				double pro;
				cin >> pro;
				Mat m2 = scale(noiseGenerator2(tmp, h, w, pro), h, w);
				imshow("Salt", m2);
				waitKey(0);
				cout << "Save the image with the file name:\n";
				string f;
				cin >> f;
				imwrite("./" + f, m2);
			}
			//��ӽ�������
			if (cmd == "3") {
				cout << "Enter the probability\n";
				double pro;
				cin >> pro;
				Mat m3 = scale(noiseGenerator3(tmp, h, w, pro), h, w);
				imshow("Salt-and-pepper", m3);
				waitKey(0);
				cout << "Save the image with the file name:\n";
				string f;
				cin >> f;
				imwrite("./" + f, m3);
			}
		}
		//�����˲�����
		if (cmd == "f") {
			cout << "1---arithmetic mean filter\n"
				<< "2---geometric mean filter\n"
				<< "3---harmonic mean filter\n"
				<< "4---contraharmonic mean filter\n"
				<< "5---max filter\n"
				<< "6---min filter\n"
				<< "7---median filter\n";
			cin >> cmd;
			//������ֵ
			if (cmd == "1") {
				cout << "Enter the size of filter,3 or 9\n";
				int len;
				cin >> len;
				if (len == 3) {
					imshow("ar3*3", scale(filter2d(tmp, h, w, ave1, 3), h, w));
				}
				else {
					imshow("ar9*9", scale(filter2d(tmp, h, w, ave2, 9), h, w));
				}
				waitKey(0);
			}
			//���ξ�ֵ
			if (cmd == "2") {
				cout << "Enter the size of filter,3 or 9\n";
				int len;
				cin >> len;
				if (len == 3) {
					r = new double[w * h];
					for (int i = 0; i < h; ++i){
						for (int j = 0; j < w; ++j){
							r[i*w + j] = calculate1(tmp, h, w, i, j, 3);
						}
					}
					imshow("geo3*3", scale(r, h, w));
				}
				else {
					r = new double[w * h];
					for (int i = 0; i < h; ++i){
						for (int j = 0; j < w; ++j){
							r[i*w + j] = calculate1(tmp, h, w, i, j, 9);
						}
					}
					imshow("geo9*9", scale(r, h, w));
				}
				waitKey(0);
			}
			//���;�ֵ
			if (cmd == "3") {
				cout << "Enter the size of filter,3 or 9\n";
				int len;
				cin >> len;
				if (len == 3) {
					for (int i = 0; i < h; i++) {
						for (int j = 0; j < w; j++) {
							tmp1[i * w + j] = 1 / tmp[i * w + j];
						}
					}
					res1 = filter2d(tmp1, h, w, ave1, 3);
					for (int i = 0; i < h; i++) {
						for (int j = 0; j < w; j++) {
							res1[i * w + j] = 1 / res1[i * w + j];
						}
					}
					imshow("h3*3", scale(res1, h, w));
				}
				else {
					for (int i = 0; i < h; i++) {
						for (int j = 0; j < w; j++) {
							tmp1[i * w + j] = 1 / tmp[i * w + j];
						}
					}
					res2 = filter2d(tmp1, h, w, ave2, 9);
					for (int i = 0; i < h; i++) {
						for (int j = 0; j < w; j++) {
							res2[i * w + j] = 1 / res2[i * w + j];
						}
					}
					imshow("h9*9", scale(res2, h, w));
				}
				waitKey(0);
			}
			//�����;�ֵ
			if (cmd == "4") {
				cout << "Enter the size of filter,3 or 9\n";
				int len;
				cin >> len;
				if (len == 3) {
					for (int i = 0; i < h; i++) {
						for (int j = 0; j < w; j++) {
							tmp1[i*w + j] = pow(tmp[i*w + j], -0.5);
							tmp2[i*w + j] = pow(tmp[i*w + j], -1.5);

						}
					}
					res1 = filter2d(tmp1, h, w, ave1, 3);
					res2 = filter2d(tmp2, h, w, ave1, 3);
					for (int i = 0; i < h; i++) {
						for (int j = 0; j < w; j++) {
							tmp3[i*w + j] = res1[i*w + j] / res2[i*w + j];
						}
					}
					imshow("ch3*3", scale(tmp3, h, w));
				}
				else {
					for (int i = 0; i < h; i++) {
						for (int j = 0; j < w; j++) {
							tmp1[i*w + j] = pow(tmp[i*w + j], -0.5);
							tmp2[i*w + j] = pow(tmp[i*w + j], -1.5);

						}
					}
					res1 = filter2d(tmp1, h, w, ave2, 9);
					res2 = filter2d(tmp2, h, w, ave2, 9);
					for (int i = 0; i < h; i++) {
						for (int j = 0; j < w; j++) {
							tmp4[i*w + j] = res1[i*w + j] / res2[i*w + j];
						}
					}
					imshow("ch9*9", scale(tmp4, h, w));
				}
				waitKey(0);
			}
			//���ֵ
			if (cmd == "5") {
				r = new double[w * h];
				for (int i = 0; i < h; ++i){
					for (int j = 0; j < w; ++j){
						r[i*w + j] = calculate4(tmp, h, w, i, j, 3);
					}
				}
				imshow("max3*3", scale(r, h, w));
				waitKey(0);
			}
			//��Сֵ
			if (cmd == "6") {
				r = new double[w * h];
				for (int i = 0; i < h; ++i){
					for (int j = 0; j < w; ++j){
						r[i*w + j] = calculate3(tmp, h, w, i, j, 3);
					}
				}
				imshow("min3*3", scale(r, h, w));
				waitKey(0);
			}
			//��ֵ
			if (cmd == "7") {
				r = new double[w * h];
				for (int i = 0; i < h; ++i){
					for (int j = 0; j < w; ++j){
						r[i*w + j] = calculate2(tmp, h, w, i, j, 3);
					}
				}
				imshow("median3*3", scale(r, h, w));
				waitKey(0);
			}
		}
		//ֱ��ͼ���⻯
		if (cmd == "h") {
			cout << "1---histogram equalization seperately\n"
				<< "2---histogram equalization averagely\n";
			cin >> cmd;
			if (cmd == "1") {
				Mat in = imread("./81.png");
				imshow("in", in);
				int h = in.rows;
				int w = in.cols;
				Mat R(h, w, CV_8UC1, Scalar(0, 0, 0)), G(h, w, CV_8UC1, Scalar(0, 0, 0)), B(h, w, CV_8UC1, Scalar(0, 0, 0));
				Mat res(h, w, CV_8UC3, Scalar(0, 0, 0));
				std::vector<cv::Mat>sbgr(in.channels());
				split(in, sbgr);
				B = sbgr[0];
				G = sbgr[1];
				R = sbgr[2];
				R = equalize_hist1(R);
				G = equalize_hist1(G);
				B = equalize_hist1(B);
				vector<Mat> v = { B, G, R };
				merge(v, res);
				imshow("res", res);
				waitKey(0);
			}
			if (cmd == "2") {
				Mat in = imread("./81.png");
				imshow("in", in);
				int h = in.rows;
				int w = in.cols;
				Mat R(h, w, CV_8UC1, Scalar(0, 0, 0)), G(h, w, CV_8UC1, Scalar(0, 0, 0)), B(h, w, CV_8UC1, Scalar(0, 0, 0));
				Mat res(h, w, CV_8UC3, Scalar(0, 0, 0));
				std::vector<cv::Mat>sbgr(in.channels());
				split(in, sbgr);
				B = sbgr[0];
				G = sbgr[1];
				R = sbgr[2];
				res = equalize_hist2(R, G, B);
				imshow("res", res);
				waitKey(0);
			}
		}
	}
	return 0;
}