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
//拉伸（使用最临近插值方法）
Mat scale1(Mat input_img, size s) {
	//获取目标大小的宽和高
	int h = s.getHeight(), w = s.getWidth();
	//新建两个单通道，8位深的图片
	Mat img1(input_img.rows, input_img.cols, CV_8UC1, Scalar(0, 0, 0)), res(h, w, CV_8UC1, Scalar(0, 0, 0));
	//将原来的三通道，8位深的图片转换为单通道的
	cvtColor(input_img, img1, CV_BGR2GRAY);
	//记录原图和目标图的宽的比以及高的比
	double wScale = input_img.cols / (w + 0.0), hScale = input_img.rows / (h + 0.0);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			//通过比例根据目标位置获取原图的像素位置
			int sourX, sourY;
			//将得到的浮点数坐标进行四舍五入，注意不能超出原图的边界
			if (ceil(i * hScale) >= input_img.rows)
				sourX = floor(i * hScale);
			else sourX = ceil(i * hScale);
			if (ceil(j * wScale) >= input_img.cols)
				sourY = floor(j * wScale);
			else sourY = ceil(j * wScale);
			//将原图的像素信息存储到目标图上
			res.at<uchar>(i, j) = img1.at<uchar>(sourX, sourY);
		}
	}
	//返回目标图
	return res;
}

//拉伸（使用双线性插值方法）
Mat scale2(Mat input_img, size s) {
	//获取目标大小的宽和高
	int h = s.getHeight(), w = s.getWidth();
	//新建两个单通道，8位深的图片
	Mat img1(input_img.rows, input_img.cols, CV_8UC1, Scalar(0, 0, 0)), res(h, w, CV_8UC1, Scalar(0, 0, 0));
	//将原来的三通道，8位深的图片转换为单通道的
	cvtColor(input_img, img1, CV_BGR2GRAY);
	//记录原图和目标图的宽的比以及高的比
	double wScale = input_img.cols / (w + 0.0), hScale = input_img.rows / (h + 0.0);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			//通过比例根据目标位置获取原图的像素位置
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
			//将原图的像素信息存储到目标图上
			res.at<uchar>(i, j) = value;
		}
	}
	//返回目标图
	return res;
}
//量化
Mat quantize(Mat input_img, int level) {
	//获取目标大小的宽和高
	int h = input_img.rows, w = input_img.cols;
	//新建两个单通道，8位深的图片
	Mat img1(h, w, CV_8UC1, Scalar(0, 0, 0)), res(h, w, CV_8UC1, Scalar(0, 0, 0));
	//将原来的三通道，8位深的图片转换为单通道的
	cvtColor(input_img, img1, CV_BGR2GRAY);
	//算出给定灰度数下的灰度值间距
	int interval = 255 / (level - 1);
	//存储灰度值的容器
	vector<int> v;
	//计算灰度值并存储
	for (int i = 0; i <= 255;){
		v.push_back(i);
		i += interval;
	}
	//扫描图像
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			//获取特定像素点的灰度值
			Scalar intensity = img1.at<uchar>(i, j);
			//遍历容器
			for (int k = 0; k < v.size() - 1; k++)
			{
				//判断灰度值落在哪个区间
				if (v[k] < intensity.val[0] && v[k + 1] / 2 > intensity.val[0]) {
					//当灰度值落在较小的半区间，把下界值赋给目标像素点
					res.at<uchar>(i, j) = v[k];
					break;
				}
				else if (v[k + 1] / 2 < intensity.val[0] && v[k + 1] > intensity.val[0]){
					//当灰度值落在较大的半区间，把上界值赋给目标像素点
					res.at<uchar>(i, j) = v[k + 1];
					break;
				}
			}
			
		}

	}
	//返回目标图
	return res;
}

int main()
{
	//读入原图
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
		//写出目标图
		imwrite("./newimage.png", img1);
	}
	return 0;
}

