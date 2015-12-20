#include <cv.h>
#include <highgui.h>
#include <math.h>
#include<iostream>
#define Pi 3.1415926
using namespace std;
using namespace cv;

//实现不同功能的滤波器
double *ave2 = new double[50];
double Lap[10] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };

//初始化滤波器
void init() {
	for (int i = 0; i < 49; i++) {
		ave2[i] = 1.0 / 49.0;
	}
}
/**
函数功能：实现傅里叶正变换和反变换
输入输出：input_img 输入的图像
          flag 标志，当标志为true时为正变换，返回的是一幅两通道的图片；
	      当标志为false时为反变换，返回的是和原图很接近的图片
*/
Mat dft2d(Mat input_img, bool flag) {
	int h = input_img.rows, w = input_img.cols;
	Mat img1(h, w, CV_8UC1, Scalar(0, 0, 0)), img2(h, w, CV_8UC2, Scalar(0, 0, 0)), spec(h, w, CV_8UC1, Scalar(0, 0, 0));
	Mat res(h, w, CV_32FC2, Scalar(0, 0, 0)), a(h, w, CV_32FC1, Scalar(0, 0, 0)), b(h, w, CV_32FC1, Scalar(0, 0, 0));
	if (input_img.channels() == 3) {
		cvtColor(input_img, img1, CV_BGR2GRAY);
	}
	else {
		img1=input_img;
	}
	//正变换
	if (flag) {
		double* image1 = new double[h * w];
		double* real1 = new double[h * w];
		double* image2 = new double[h * w];
		double* real2 = new double[h * w];
		double max = -10000000, min = 10000000, maxRe = -10000000, maxIm = -10000000, minRe = 10000000, minIm = 10000000;
		Mat res(h, w, CV_8UC2, Scalar(0, 0, 0)), Im(h, w, CV_8UC1, Scalar(0, 0, 0)), Re(h, w, CV_8UC1, Scalar(0, 0, 0));
		for (int i = 0; i < h * w; i++) {
			image1[i] = real1[i] = image2[i] = real2[i] = 0;
		}
		int cy = h / 2; // image center  
		int cx = w / 2;
		double tmp = 0;
		double scaleRe, shiftRe, scaleIm, shiftIm, scale, shift;
		//进行y方向的1-d DFT
		for (int v = 0; v < w; v++) {
			for (int x = 0; x < h; x++){
				for (int y = 0; y < w; y++) {
					Scalar intensity = img1.at<uchar>(x, y);
					real1[x * w + v] += intensity.val[0] * cos(2 * Pi * v * y / w);
					image1[x * w + v] += intensity.val[0] * (-sin(2 * Pi * v * y / w));
				}
			}
		}
		//进行x方向的1-d DFT
		for (int u = 0; u < h; u++) {
			for (int v = 0; v < w; v++) {
				for (int x = 0; x < h; x++) {
					real2[u * w + v] += cos(2 * Pi * u * x / h) * real1[x * w + v] +
						sin(2 * Pi * u * x / h) * image1[x * w + v];
					image2[u * w + v] += -sin(2 * Pi * u * x / h) * real1[x * w + v] +
						cos(2 * Pi * u * x / h) * image1[x * w + v];
				}
			}
		}
		//计算谱的大小并对谱进行log变换
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				spec.at<uchar>(i, j) = log(1.0 + sqrt(pow(real2[i*w + j], 2) + pow(image2[i*w + j], 2)));
				b.at<float>(i, j) = image2[i * w + j];
				a.at<float>(i, j) = real2[i * w + j];
			}
		}
		//将实部和虚部合并为一张两通道的图片
		Mat planes[] = { a, b };
		merge(planes, 2, res);
		//标定
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				Scalar tmp = spec.at<uchar>(i, j);
				if (max < tmp.val[0]) max = tmp.val[0];
				if (min > tmp.val[0]) min = tmp.val[0];
			}
		}
		scale = 255 / (max - min);
		shift = -min * scale;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				Scalar tmp = spec.at<uchar>(i, j);
				spec.at<uchar>(i, j) = scale * tmp.val[0] + shift;
			}
		}
		//中心化
		for (int j = 0; j < cy; j++){
			for (int i = 0; i < cx; i++){
				Scalar t = spec.at<uchar>(j, i);
				tmp = t.val[0];
				spec.at<uchar>(j, i) = spec.at<uchar>(j + cy, i + cx);
				spec.at<uchar>(j + cy, i + cx) = tmp;
				t = spec.at<uchar>(j, i + cx);
				tmp = t.val[0];
				spec.at<uchar>(j, i + cx) = spec.at<uchar>(j + cy, i);
				spec.at<uchar>(j + cy, i) = tmp;
			}
		}
		//保存图片
		imshow("spec", spec);
		//imwrite("./spec.png", spec);
		waitKey(0);
		return res;
	}
	else {
		double* image1 = new double[h * w];
		double* real1 = new double[h * w];
		double* image2 = new double[h * w];
		double* real2 = new double[h * w];
		double max = -10000000, min = 10000000;
		Mat res(h, w, CV_8UC1, Scalar(0, 0, 0)), Im(h, w, CV_8UC1, Scalar(0, 0, 0)), Re(h, w, CV_8UC1, Scalar(0, 0, 0));
		for (int i = 0; i < h * w; i++) {
			image1[i] = real1[i] = image2[i] = real2[i] = 0;
		}
		Mat a(h, w, CV_32FC1, Scalar(0, 0, 0)), b(h, w, CV_32FC1, Scalar(0, 0, 0));
		Mat planes[] = { a, b };
		split(input_img, planes);
		double scale, shift;
		int cy = h / 2; // image center  
		int cx = w / 2;
		double tmp = 0;
		//进行v方向的1-d DFT
		for (int y = 0; y < w; y++) {
			for (int u = 0; u < h; u++){
				for (int v = 0; v < w; v++) {
					//Scalar re = Re.at<uchar>(u, v), im = Im.at<uchar>(u, v);
					real1[u * w + y] += a.at<float>(u, v) * cos(2 * Pi * v * y / w) -
						b.at<float>(u, v) * sin(2 * Pi * v * y / w);
					image1[u * w + y] += a.at<float>(u, v) * sin(2 * Pi * v * y / w) +
						b.at<float>(u, v) * cos(2 * Pi * v * y / w);
				}
			}
		}
		//进行u方向的1-d DFT
		for (int x = 0; x < h; x++) {
			for (int y = 0; y < w; y++) {
				for (int u = 0; u < h; u++) {
					real2[x * w + y] += (cos(2 * Pi * u * x / h) * real1[u * w + y] -
						sin(2 * Pi * u * x / h) * image1[u * w + y]);
				}
			}
		}
		//标定
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				real2[i * w + j] = real2[i * w + j] / (h * w);
				if (max < real2[i * w + j]) max = real2[i * w + j];
				if (min > real2[i * w + j]) min = real2[i * w + j];
			}
		}
		scale = 255 / (max - min);
		shift = -min * scale;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				res.at<uchar>(i, j) = scale * real2[i * w + j] + shift;
			}
		}
		//imshow("res", res);
		//waitKey(0);
		//imwrite("./res.png", res);
		return res;
	}
}
/**
函数功能：实现频率域的滤波
输入：input_img 输入图片
          filter  空间域的滤波器
		  size  滤波器的大小
输出：经过处理后的图片
*/
Mat filter2d_freq(Mat input_img, double* filter, int size){
	int h = input_img.rows, w = input_img.cols;
	if (input_img.channels() == 3) {
		cvtColor(input_img, input_img, CV_BGR2GRAY);
	}
	Mat F(2 * h, 2 * w, CV_32FC2, Scalar(0, 0, 0)), H(2 * h, 2 * w, CV_32FC2, Scalar(0, 0, 0)), G(2 * h,2 * w, CV_32FC2, Scalar(0, 0, 0));
	Mat hx(2 * h, 2 * w, CV_8UC1, Scalar(0, 0, 0)), input(2 * h, 2 * w, CV_8UC1, Scalar(0, 0, 0)), res(2 * h, 2 * w, CV_8UC1, Scalar(0, 0, 0)), 
		r(h, w, CV_8UC1, Scalar(0, 0, 0));
	//对原图像和滤波器进行零填充，使得填充后的大小相同，这里设置的长宽为2*M和2*N
	for (int i = 0; i < 2 * h; i++) {
		for (int j = 0; j < 2 * w; j++) {
			if (i < size && j < size) {
				hx.at<uchar>(i, j) = filter[i * size + j];
			}
			else {
				hx.at<uchar>(i, j) = 0;
			}
		}
	}
	for (int i = 0; i < 2 * h; i++) {
		for (int j = 0; j < 2 * w; j++) {
			if (i < h && j < w) {
				input.at<uchar>(i, j) = input_img.at<uchar>(i, j);
			}
			else {
				input.at<uchar>(i, j) = 0;
			}
		}
	}
	int cy = h, cx = w;
	double tmp;
	double max = -10000000, min = 10000000, scale, shift;
	//将原图像和滤波器进行正变换
	F = dft2d(input, true);
	H = dft2d(hx, true);
	Mat Re1(2 * h, 2 * w, CV_32FC1, Scalar(0, 0, 0)), Re2(2 * h, 2 * w, CV_32FC1, Scalar(0, 0, 0)), Re3(2 * h, 2 * w, CV_32FC1, Scalar(0, 0, 0)),
		Im1(2 * h, 2 * w, CV_32FC1, Scalar(0, 0, 0)), Im2(2 * h, 2 * w, CV_32FC1, Scalar(0, 0, 0)), Im3(2 * h, 2 * w, CV_32FC1, Scalar(0, 0, 0));
	//将实部虚部分离
	Mat planes1[] = {Re1, Im1};
	split(F, planes1);
	Mat planes2[] = { Re2, Im2 };
	split(H, planes2);
	//H(u,v)*F(u,v)
	for (int i = 0; i < 2 * h; i++) {
		for (int j = 0; j < 2 * w; j++) {
			Re3.at<float>(i, j) = Re1.at<float>(i, j) * Re2.at<float>(i, j) - Im1.at<float>(i, j) * Im2.at<float>(i, j);
			Im3.at<float>(i, j) = Re1.at<float>(i, j) * Im2.at<float>(i, j) + Im1.at<float>(i, j) * Re2.at<float>(i, j);

		}
	}
	//得到处理后的G(u,v)
	Mat planes[] = {Re3, Im3};
	merge(planes, 2, G);
	//进行逆变换
	res = dft2d(G, false);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			r.at<uchar>(i, j) = res.at<uchar>(i, j);
		}
	}
	//imshow("filtered res", r);
	//imwrite("./filtered res.png", r);
	//waitKey(0);
	return r;
}

int main(){
	init();
	//读入原图
	Mat img = imread("./81.png");
	string cmd, cmd1;
	while (true) {
		Mat img1(img.rows, img.cols, CV_8UC1, Scalar(0, 0, 0));
		Mat smooth(img.rows, img.cols, CV_8UC1, Scalar(0, 0, 0)), sharpen(img.rows, img.cols, CV_8UC1, Scalar(0, 0, 0));
		cout << "dft----Fourier Transform\n"
			<< "ff----Frequency Filtering\n";
		cin >> cmd;
		if (cmd == "dft") {
			imshow("source image", img);
			img1 = dft2d(img, true);
			imshow("after idft", dft2d(img1, false));
			waitKey(0);
		}
		if (cmd == "ff") {
			cout << "1----Smooth your input image\n"
				<< "2----Sharpen your input image\n";
			cin >> cmd1;
			if (cmd1 == "1") {
				smooth = filter2d_freq(img, ave2, 7);
				imshow("smooth", smooth);
				waitKey(0);
			}
			if (cmd1 == "2") {
				sharpen = filter2d_freq(img, Lap, 3);
				imshow("sharpen", sharpen);
				waitKey(0);
			}
		}
	}
	return 0;
}

