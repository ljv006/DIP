#include <cv.h>
#include <highgui.h>
using namespace std;
using namespace cv;


double* ave1 = new double[10], *ave2 = new double[50], *ave3 = new double[122];
double Lap[10] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
double K[10] = { 0, 0, 0, 0, 1.2, 0, 0, 0, 0 };
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
Mat print_hist(Mat img) {
	int bins = 256;
	int hist[256];
	for (int i = 0; i < 256; i++) {
		hist[i] = 0;
	}
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Scalar intensity = img.at<uchar>(i, j);
			int tmp = intensity.val[0];
			hist[tmp] += 1; //��Ӧ�Ҷ�ֵ���ص���������һ
		}
	}
	double max_val = 0;
	for (int i = 0; i < 256; i++) {
		if (hist[i] > max_val)
			max_val = hist[i];
	}

	int scale = 2;
	int hist_height = 256;
	Mat hist_img = Mat::zeros(hist_height, bins*scale, CV_8UC3);
	for (int i = 0; i < bins; i++) {
		float bin_val = hist[i];
		int intensity = cvRound(bin_val*hist_height / max_val);  //Ҫ���Ƶĸ߶�  
		rectangle(hist_img, Point(i*scale, hist_height - 1),
			Point((i + 1)*scale - 1, hist_height - intensity),
			CV_RGB(255, 255, 255));
	}
	return hist_img;
}
Mat equalize_hist(Mat input_img) {
	//��ȡĿ���С�Ŀ�͸�
	int h = input_img.rows, w = input_img.cols;
	//�½�������ͨ����8λ���ͼƬ
	Mat img1(h, w, CV_8UC1, Scalar(0, 0, 0)), res(h, w, CV_8UC1, Scalar(0, 0, 0));
	//��ԭ������ͨ����8λ���ͼƬת��Ϊ��ͨ����
	if (input_img.channels() == 3) {
		cvtColor(input_img, img1, CV_BGR2GRAY);
	}
	else {
		img1 = input_img;
	}
	Mat hist_img1, hist_res;
	hist_img1 = print_hist(img1).clone();
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
int calculate(Mat input_img, double* filter, int x, int y, int len) {
	int sum = 0;
	int h = input_img.rows, w = input_img.cols;
	for (int i = 0; i < len; ++i){
		for (int j = 0; j < len; ++j){
			if (((x - 1 + i) >= 0) && ((y - 1 + j) >= 0) && ((x - 1 + i) < h) && ((y - 1 + j) < w)){
				Scalar intensity = input_img.at<uchar>(x - 1 + i,y - 1 + j);
				sum = sum + filter[len * i + j] * intensity.val[0];
			}
		}
	}
	if (sum > 255) sum = 255;
	if (sum < 0) sum = 0;
	return sum;
}

Mat filter2d(Mat input_img, double* filter, int len) {
	// ��ȡĿ���С�Ŀ�͸�
	int h = input_img.rows, w = input_img.cols;
	//�½�������ͨ����8λ���ͼƬ
	Mat img1(h, w, CV_8UC1, Scalar(0, 0, 0)), res(h, w, CV_8UC1, Scalar(0, 0, 0));
	//��ԭ������ͨ����8λ���ͼƬת��Ϊ��ͨ����
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
	init();
	Mat img = imread("./81.png");
	//equalize_hist(img);
	imshow("Source", img);
	//imshow("3*3", filter2d(img, ave1, 3));
	//imshow("7*7", filter2d(img, ave2, 7));
	//imshow("11*11", filter2d(img, ave3, 11));
	//imshow("Sharpen", filter2d(img, Lap, 3));
	Mat temp(img.rows, img.cols, CV_8UC1, Scalar(0, 0, 0)), highBoost(img.rows, img.cols, CV_8UC1, Scalar(0, 0, 0));
	Mat img1(img.rows, img.cols, CV_8UC1, Scalar(0, 0, 0));
	//��ԭ������ͨ����8λ���ͼƬת��Ϊ��ͨ����
	cvtColor(img, img1, CV_BGR2GRAY);
	subtract(img1, filter2d(img1, ave1, 3), temp);
	add(img1, filter2d(temp, K, 3), highBoost);
	imshow("Highboost1.2", highBoost);
	waitKey(10000000000);
	return 0;
}