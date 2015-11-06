/****************************************************************************************/
/*     Author:longjiawei                                                                */
/* This problem can realize histogram equalization and spatial filtering of a picture.  */
/****************************************************************************************/
#include <cv.h>
#include <highgui.h>
using namespace std;
using namespace cv;

//ʵ�ֲ�ͬ���ܵ��˲���
double* ave1 = new double[10], *ave2 = new double[50], *ave3 = new double[122];
double Lap[10] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
double K[10] = { 0, 0, 0, 0, 1.2, 0, 0, 0, 0 };
double test[10] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
//��ʼ���˲���
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
//��ӡֱ��ͼ
Mat print_hist(Mat img) {
	//ֱ��ͼ�ĵ�Ԫ��
	int bins = 256;
	//��ʼ��ֱ��ͼ
	int hist[256];
	for (int i = 0; i < 256; i++) {
		hist[i] = 0;
	}
	//ͳ��ÿ���Ҷ�ֵ������
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Scalar intensity = img.at<uchar>(i, j);
			int tmp = intensity.val[0];
			hist[tmp] += 1; //��Ӧ�Ҷ�ֵ���ص���������һ
		}
	}
	//ȷ�����ݵ����ֵ��ȷ��ֱ��ͼ�ĸ߶�
	double max_val = 0;
	for (int i = 0; i < 256; i++) {
		if (hist[i] > max_val)
			max_val = hist[i];
	}
	//ʹ��API����ֱ��ͼ��ӡ
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
	//����ֱ��ͼ
	return hist_img;
}
//ֱ��ͼ���⻯
Mat equalize_hist(Mat input_img) {
	//��ȡĿ���С�Ŀ�͸�
	int h = input_img.rows, w = input_img.cols;
	//�½�������ͨ����8λ���ͼƬ,һ��Ϊԭͼת�ɵ�ͨ�����ͼ����һ��Ϊ�����Ľ��
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
//�������
//���ڱ�Ե���������˸ı���س��˲�����С�Ĳ���
int calculate(Mat input_img, double* filter, int x, int y, int len) {
	int sum = 0;
	int h = input_img.rows, w = input_img.cols;
	for (int i = 0; i < len; ++i){
		for (int j = 0; j < len; ++j){
			//�ж��Ƿ񳬳���ͼƬ�ı߽�
			if (((x - 1 + i) >= 0) && ((y - 1 + j) >= 0) && ((x - 1 + i) < h) && ((y - 1 + j) < w)){
				Scalar intensity = input_img.at<uchar>(x - 1 + i,y - 1 + j);
				sum = sum + filter[len * i + j] * intensity.val[0];
			}
		}
	}
	//�ж�����ֵ�Ƿ񳬳���0-255�ķ�Χ
	if (sum > 255) sum = 255;
	if (sum < 0) sum = 0;
	return sum;
}
//�����˲�������ͼ����д���
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
	//��ʼ���˲���
	init();
	//����ԭͼƬ
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
				//��ԭͼ��ȥƽ�����ͼ���õ�ģ
				subtract(img1, filter2d(img1, ave1, 3), temp);
				//��ģ����һ��ϵ��������ԭͼ��ӣ��õ���Ե��ǿ��ͼ
				//�˴���ϵ��������1.2
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