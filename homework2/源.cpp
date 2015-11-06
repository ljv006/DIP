/****************************************************************************************/
/*     Author:longjiawei                                                                */
/* This problem can realize histogram equalization and spatial filtering of a picture.  */
/****************************************************************************************/
#include <cv.h>
#include <highgui.h>
using namespace std;
using namespace cv;

//实现不同功能的滤波器
double* ave1 = new double[10], *ave2 = new double[50], *ave3 = new double[122];
double Lap[10] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
double K[10] = { 0, 0, 0, 0, 1.2, 0, 0, 0, 0 };
double test[10] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
//初始化滤波器
void init() {
	for (int i = 0; i < 9; i++) {
			ave1[i] = 1.0 / 9.0;
	}
	for (int i = 0; i < 49; i++) {
			ave2[i] = 1.0 / 49.0;
	}
	for (int i = 0; i < 121; i++) {
			ave3[i] = 1.0 / 121.0;
	}
}
//打印直方图
Mat print_hist(Mat img) {
	//直方图的单元数
	int bins = 256;
	//初始化直方图
	int hist[256];
	for (int i = 0; i < 256; i++) {
		hist[i] = 0;
	}
	//统计每个灰度值的数量
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Scalar intensity = img.at<uchar>(i, j);
			int tmp = intensity.val[0];
			hist[tmp] += 1; //对应灰度值像素点数量增加一
		}
	}
	//确定数据的最大值已确定直方图的高度
	double max_val = 0;
	for (int i = 0; i < 256; i++) {
		if (hist[i] > max_val)
			max_val = hist[i];
	}
	//使用API进行直方图打印
	int scale = 2;
	int hist_height = 256;
	Mat hist_img = Mat::zeros(hist_height, bins*scale, CV_8UC3);
	for (int i = 0; i < bins; i++) {
		float bin_val = hist[i];
		int intensity = cvRound(bin_val*hist_height / max_val);  //要绘制的高度  
		rectangle(hist_img, Point(i*scale, hist_height - 1),
			Point((i + 1)*scale - 1, hist_height - intensity),
			CV_RGB(255, 255, 255));
	}
	//返回直方图
	return hist_img;
}
//直方图均衡化
Mat equalize_hist(Mat input_img) {
	//获取目标大小的宽和高
	int h = input_img.rows, w = input_img.cols;
	//新建两个单通道，8位深的图片,一个为原图转成单通道后的图，另一个为处理后的结果
	Mat img1(h, w, CV_8UC1, Scalar(0, 0, 0)), res(h, w, CV_8UC1, Scalar(0, 0, 0));
	//将原来的三通道，8位深的图片转换为单通道的
	if (input_img.channels() == 3) {
		cvtColor(input_img, img1, CV_BGR2GRAY);
	}
	else {
		img1 = input_img;
	}
	Mat hist_img1, hist_res;
	hist_img1 = print_hist(img1).clone();
	//进行直方图均衡化
	int NumPixel[256]; //统计各灰度数目，共256个灰度级
	for (int i = 0; i < 256; i++) {
		NumPixel[i] = 0;
	}
	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			Scalar intensity = img1.at<uchar>(i, j);
			int tmp = intensity.val[0];
			NumPixel[tmp] += 1; //对应灰度值像素点数量增加一
		}
	}
	//计算灰度分布密度
	double ProbPixel[256];
	for (int i = 0; i < 256; i++) {
		ProbPixel[i] = 0;
	}
	for (int i = 0; i < 256; i++) {
		ProbPixel[i] = NumPixel[i] / (img1.cols * img1.rows * 1.0);
	}
	//计算累计直方图分布
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
	//累计分布取整
	for (int i = 0; i < 256; i++) {
		CumuPixel[i] = 255 * CumuPixel[i] + 0.5;
	}
	
	//对灰度值进行映射（均衡化）
	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			res.at<uchar>(i, j) = CumuPixel[img1.at<uchar>(i, j)];
		}
	}
	imshow("Source", img1);
	imshow("Source Histogram", hist_img1);
	imshow("Result", res);
	hist_res = print_hist(res).clone();
	imshow("Result Histogram", hist_res);
	waitKey(10000000000);
	imwrite("./Source.png", img1);
	imwrite("./SourceHist.png", hist_img1);
	imwrite("./Result.png", res);
	imwrite("./ResultHist.png", hist_res);
	return res;
}
//卷积运算
//对于边缘处理，采用了改变边沿出滤波器大小的策略
int calculate(Mat input_img, double* filter, int x, int y, int len) {
	int sum = 0;
	int h = input_img.rows, w = input_img.cols;
	for (int i = 0; i < len; ++i){
		for (int j = 0; j < len; ++j){
			//判断是否超出了图片的边界
			if (((x - 1 + i) >= 0) && ((y - 1 + j) >= 0) && ((x - 1 + i) < h) && ((y - 1 + j) < w)){
				Scalar intensity = input_img.at<uchar>(x - 1 + i,y - 1 + j);
				sum = sum + filter[len * i + j] * intensity.val[0];
			}
		}
	}
	//判断像素值是否超出了0-255的范围
	if (sum > 255) sum = 255;
	if (sum < 0) sum = 0;
	return sum;
}
//接收滤波器并对图像进行处理
Mat filter2d(Mat input_img, double* filter, int len) {
	// 获取目标大小的宽和高
	int h = input_img.rows, w = input_img.cols;
	//新建两个单通道，8位深的图片
	Mat img1(h, w, CV_8UC1, Scalar(0, 0, 0)), res(h, w, CV_8UC1, Scalar(0, 0, 0));
	//将原来的三通道，8位深的图片转换为单通道的
	if (input_img.channels() == 3) {
		cvtColor(input_img, img1, CV_BGR2GRAY);
	}
	else {
		img1 = input_img;
	}
	for (int i = 0; i < h; ++i){
		for (int j = 0; j < w; ++j){
			res.at<uchar>(i,j) = calculate(img1, filter, i, j, len);
		}
	}
	return res;
}

int main()
{
	//初始化滤波器
	init();
	//读入原图片
	Mat img = imread("./81.png");
	string cmd, cmd1;
	while (true) {
		Mat img1(img.rows, img.cols, CV_8UC1, Scalar(0, 0, 0));
		Mat temp(img.rows, img.cols, CV_8UC1, Scalar(0, 0, 0)), highBoost(img.rows, img.cols, CV_8UC1, Scalar(0, 0, 0));
		cout << "h----Histogram Equalization\n"
			<< "s----Spatial Filtering\n";
		cin >> cmd;
		if (cmd == "h") {
			equalize_hist(img);
		}
		if (cmd == "s") {
			cout << "1----Smooth your input image\n"
				<< "2----Sharpen your input image\n"
				<< "3----High-boost filtering\n";
			cin >> cmd1;
			if (cmd1 == "1") {
				cout << "Entering the size of averaging filter.\n";
				int len;
				cin >> len;
				if (len == 3) {
					imshow("Source", img);
					imshow("3*3", filter2d(img, ave1, 3));
					waitKey(0);
					imwrite("./3.png", filter2d(img, ave1, 3));
				}
				else if (len == 7) {
					imshow("Source", img);
					imshow("7*7", filter2d(img, ave2, 7));
					waitKey(0);
					imwrite("./7.png", filter2d(img, ave2, 7));
				}
				else if (len == 11) {
					imshow("Source", img);
					imshow("11*11", filter2d(img, ave3, 11));
					waitKey(0);
					imwrite("./11.png", filter2d(img, ave3, 11));
				}
				else if (len == 12) {
					imshow("Source", img);
					imshow("test", filter2d(img, test, 3));
					waitKey(0);
					imwrite("./test.png", filter2d(img, test, 3));
				}
			}
			if (cmd1 == "2") {
				imshow("Source", img);
				imshow("Sharpen", filter2d(img, Lap, 3));
				waitKey(0);
				imwrite("./Sharpen.png", filter2d(img, Lap, 3));
			}
			if (cmd1 == "3") {
				cvtColor(img, img1, CV_BGR2GRAY);
				//用原图减去平滑后的图，得到模
				subtract(img1, filter2d(img1, ave1, 3), temp);
				//将模乘上一个系数后，再与原图相加，得到边缘增强的图
				//此处的系数采用了1.2
				add(img1, filter2d(temp, K, 3), highBoost);
				imshow("Source", img);
				imshow("Highboost", highBoost);
				waitKey(0);
				imwrite("./Highboost.png", highBoost);
			}
		}
	}
	return 0;
}