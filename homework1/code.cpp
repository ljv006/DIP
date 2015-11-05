#include <cv.h>
#include <highgui.h>
#include <vector>
#include <math.h>
#include<iostream>

using namespace std;
using namespace cv;
class size{
public:
	size(int w = 0, int h = 0):width(w),height(h) {
	}
	int getWidth() {
		return width;
	}
	int getHeight() {
		return height;
	}
private:
	int width;
	int height;
};
//���죨ʹ�����ٽ���ֵ������
Mat scale1(Mat input_img, size s) {
	//��ȡĿ���С�Ŀ�͸�
	int h = s.getHeight(), w = s.getWidth();
	//�½�������ͨ����8λ���ͼƬ
	Mat img1(input_img.rows, input_img.cols, CV_8UC1, Scalar(0, 0, 0)), res(h, w, CV_8UC1, Scalar(0, 0, 0));
	//��ԭ������ͨ����8λ���ͼƬת��Ϊ��ͨ����
	cvtColor(input_img, img1, CV_BGR2GRAY);
	//��¼ԭͼ��Ŀ��ͼ�Ŀ�ı��Լ��ߵı�
	double wScale = input_img.cols / (w + 0.0), hScale = input_img.rows / (h + 0.0);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			//ͨ����������Ŀ��λ�û�ȡԭͼ������λ��
			int sourX, sourY;
			//���õ��ĸ�������������������룬ע�ⲻ�ܳ���ԭͼ�ı߽�
			if (ceil(i * hScale) >= input_img.rows)
				sourX = floor(i * hScale);
			else sourX = ceil(i * hScale);
			if (ceil(j * wScale) >= input_img.cols)
				sourY = floor(j * wScale);
			else sourY = ceil(j * wScale);
			//��ԭͼ��������Ϣ�洢��Ŀ��ͼ��
			res.at<uchar>(i, j) = img1.at<uchar>(sourX, sourY);
		}
	}
	//����Ŀ��ͼ
	return res;
}

//���죨ʹ��˫���Բ�ֵ������
Mat scale2(Mat input_img, size s) {
	//��ȡĿ���С�Ŀ�͸�
	int h = s.getHeight(), w = s.getWidth();
	//�½�������ͨ����8λ���ͼƬ
	Mat img1(input_img.rows, input_img.cols, CV_8UC1, Scalar(0, 0, 0)), res(h, w, CV_8UC1, Scalar(0, 0, 0));
	//��ԭ������ͨ����8λ���ͼƬת��Ϊ��ͨ����
	cvtColor(input_img, img1, CV_BGR2GRAY);
	//��¼ԭͼ��Ŀ��ͼ�Ŀ�ı��Լ��ߵı�
	double wScale = input_img.cols / (w + 0.0), hScale = input_img.rows / (h + 0.0);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			//ͨ����������Ŀ��λ�û�ȡԭͼ������λ��
			int sourX, sourY;
			sourX = floor(i * hScale);
			sourY = floor(j * wScale);
			double u, v;
			u = i * hScale - sourX;
			v = j * wScale - sourY;
			Scalar intensity1 = img1.at<uchar>(sourX, sourY);
			Scalar intensity2 = img1.at<uchar>(sourX, sourY + 1);
			Scalar intensity3 = img1.at<uchar>(sourX + 1, sourY);
			Scalar intensity4 = img1.at<uchar>(sourX + 1, sourY + 1);
			int value = (1 - u)*(1 - v)*intensity1.val[0] + (1 - u)*v*intensity2.val[0] 
				+ u * (1 - v) * intensity3.val[0] + u * v * intensity4.val[0];
			//��ԭͼ��������Ϣ�洢��Ŀ��ͼ��
			res.at<uchar>(i, j) = value;
		}
	}
	//����Ŀ��ͼ
	return res;
}
//����
Mat quantize(Mat input_img, int level) {
	//��ȡĿ���С�Ŀ�͸�
	int h = input_img.rows, w = input_img.cols;
	//�½�������ͨ����8λ���ͼƬ
	Mat img1(h, w, CV_8UC1, Scalar(0, 0, 0)), res(h, w, CV_8UC1, Scalar(0, 0, 0));
	//��ԭ������ͨ����8λ���ͼƬת��Ϊ��ͨ����
	cvtColor(input_img, img1, CV_BGR2GRAY);
	//��������Ҷ����µĻҶ�ֵ���
	int interval = 255 / (level - 1);
	//�洢�Ҷ�ֵ������
	vector<int> v;
	//����Ҷ�ֵ���洢
	for (int i = 0; i <= 255;){
		v.push_back(i);
		i += interval;
	}
	//ɨ��ͼ��
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			//��ȡ�ض����ص�ĻҶ�ֵ
			Scalar intensity = img1.at<uchar>(i, j);
			//��������
			for (int k = 0; k < v.size() - 1; k++)
			{
				//�жϻҶ�ֵ�����ĸ�����
				if (v[k] < intensity.val[0] && v[k + 1] / 2 > intensity.val[0]) {
					//���Ҷ�ֵ���ڽ�С�İ����䣬���½�ֵ����Ŀ�����ص�
					res.at<uchar>(i, j) = v[k];
					break;
				}
				else if (v[k + 1] / 2 < intensity.val[0] && v[k + 1] > intensity.val[0]){
					//���Ҷ�ֵ���ڽϴ�İ����䣬���Ͻ�ֵ����Ŀ�����ص�
					res.at<uchar>(i, j) = v[k + 1];
					break;
				}
			}
			
		}

	}
	//����Ŀ��ͼ
	return res;
}

int main()
{
	//����ԭͼ
	Mat img = imread("./81.png");
	Mat img1;
	string cmd;
	while (true) {
		cout << "s1----scale your picture in nearest interpolation\n"
			<< "s2----scale your picture in bilinear interpolation\n"
			<< "q----quantization your picture\n";
		cin >> cmd;
		if (cmd == "s1") {
			int w, h;
			cout << "Entering the size.\n";
			cout << "The width:\n";
			cin >> w;
			cout << "The height:\n";
			cin >> h;
			size s = size(w, h);
			img1 = scale1(img, s);
		}
		if (cmd == "s2") {
			int w, h;
			cout << "Entering the size.\n";
			cout << "The width:\n";
			cin >> w;
			cout << "The height:\n";
			cin >> h;
			size s = size(w, h);
			img1 = scale2(img, s);
		}
		if (cmd == "q") {
			int level;
			cout << "Entering the gray level.\n";
			cout << "The level:\n";
			cin >> level;
			img1 = quantize(img, level);
		}
		namedWindow("oldphoto", CV_WINDOW_AUTOSIZE);
		namedWindow("newphoto", CV_WINDOW_AUTOSIZE);
		imshow("oldphoto", img);
		imshow("newphoto", img1);
		waitKey(0);
		//д��Ŀ��ͼ
		imwrite("./newimage.png", img1);
	}
	return 0;
}

