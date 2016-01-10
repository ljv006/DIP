#include <cv.h>
#include <highgui.h>
#include <math.h>
#define dmax 79
#define K 0.8
#define rc 13
#define rp 31
#define winSize 7
using namespace std;
using namespace cv;

string filename[21] = { "Aloe", "Baby1", "Baby2", "Baby3", "Bowling1", "Bowling2", "Cloth1",
"Cloth2", "Cloth3", "Cloth4", "Flowerpots", "Lampshade1", "Lampshade2", "Midd1", "Midd2",
"Monopoly", "Plastic", "Rocks1", "Rocks2", "Wood1", "Wood2" };
/**�������ܣ��������������ϵ��3����ת��ΪͼƬ��ʽ��
���룺input_img ��������,�洢���ص���Ϣ
h ͼƬ�ĸ߶�
w ͼƬ�Ŀ��
������궨ת�����ͼƬ
*/
Mat scale(int* input_img, int h, int w) {
	Mat tmp(h, w, CV_8UC1, Scalar(0, 0, 0));
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			tmp.at<uchar>(i, j) = 3 * input_img[i * w + j];
		}
	}
	return tmp;
}
/**�������ܣ�ͨ���Ƚϼ���õ����Ӳ�ͼ������Ӳ�ͼ������ֵ��
��¼������صĸ���������ٷֱȡ�
���룺myDisMap ͼ��������Դ�ű��������õ����Ӳ�ͼ
DisMap ͼ��������Դ�Ÿ������Ӳ�ͼ
������õ���������ذٷֱ�
*/
double evaluator(Mat myDisMap, Mat DisMap) {
	int h = myDisMap.rows;
	int w = myDisMap.cols;
	int error = 0;
	double res;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			Scalar myDM = myDisMap.at<uchar>(i, j);
			Scalar DM = DisMap.at<uchar>(i, j);
			//���������Ӳ�ͼ��������3�����Բ�ֵӦ��Ϊ3
			if (abs(myDM.val[0] - DM.val[0]) > 3) {
				error++;
			}
		}
	}
	res = (error + 0.0) / (h * w + 0.0);
	return res;
}
/**�������ܣ���������ͼ��Ӧ���ڵĶ�Ӧ���صĲ��ƽ���ĺ�
��Ϊƥ����ۣ�ѡ��������С���Ӳ��Ϊ�Ӳ�ͼ��Ӧ�������ֵ��
���룺IL��IR ������ͼ
h��w ͼƬ�ĳ���
flag ��ǣ���������������Ӳ�ͼ�������Ӳ�ͼ
��������ش���������������洢���Ӳ�ͼ
*/
int* SSD(double* IL, double* IR, int h, int w, char flag) {
	int* res = new int[h * w];
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int pos = 0;
			double min = 100000000000000.0;
			for (int k = 0; k <= dmax; k++) {
				double sum = 0;
				if (flag == 'l') {
					for (int l = i - (winSize - 1) / 2; l <= i + (winSize - 1) / 2; l++) {
						for (int m = j - (winSize - 1) / 2; m <= j + (winSize - 1) / 2; m++) {
							if (l < 0 || m - k < 0 || m < 0 || m >= w || l >= h) {
								continue;
							}
							else {
								sum += (IL[l * w + m] - IR[l * w + m - k]) * (IL[l * w + m] - IR[l * w + m - k]);
							}
						}
					}
				}
				else {
					for (int l = i - (winSize - 1) / 2; l <= i + (winSize - 1) / 2; l++) {
						for (int m = j - (winSize - 1) / 2; m <= j + (winSize - 1) / 2; m++) {
							if (l < 0 || m < 0 || m + k >= w || m >= w || l >= h) {
								continue;
							}
							else {
								sum += (IL[l * w + m + k] - IR[l * w + m]) * (IL[l * w + m + k] - IR[l * w + m]);
							}
						}
					}
				}
				if (sum < min && abs(sum-0) > 0.000000000001) {
					min = sum;
					pos = k;
				}
			}
			res[i * w + j] = pos;
		}
	}
	return res;
}
/**�������ܣ��������������ͼ���д�����NCC�ķ�ʽ�õ��������Ӳ�ͼ��
���룺IL��IR ������ͼ
h��w ͼƬ�ĳ���
flag ��ǣ���������������Ӳ�ͼ�������Ӳ�ͼ
��������ش���������������洢���Ӳ�ͼ
*/
int* NCC(double* IL, double* IR, int h, int w, char flag) {
	int* res = new int[h * w];
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int pos = 0;
			double max = -100000000000000.0;
			for (int k = 0; k <= dmax; k++) {
				double sum = 0;
				double avgR = 0, avgL = 0;
				double devR = 0, devL = 0;
				double Srl = 0, Sr = 0, Sl = 0, Srr = 0, Sll = 0;
				if (flag == 'l') {
					for (int l = i - (winSize - 1) / 2; l <= i + (winSize - 1) / 2; l++) {
						for (int m = j - (winSize - 1) / 2; m <= j + (winSize - 1) / 2 ; m++) {
							if (l < 0 || m < 0 || m - k < 0 || m >= w || l >= h) {
								continue;
							}
							else {
								Srl += IR[l * w + m - k] * IL[l * w + m];
								Sr += IR[l * w + m - k];
								Sl += IL[l * w + m];
								Srr += IR[l * w + m - k] * IR[l * w + m - k];
								Sll += IL[l * w + m] * IL[l * w + m];
							}
						}
					}
					sum = (Srl - Sr * Sl / (winSize * winSize)) / sqrt((Srr - Sr * Sr / (winSize * 
						winSize)) * (Sll - Sl * Sl / (winSize * winSize)));
				}
				else {
					for (int l = i - (winSize - 1) / 2; l <= i + (winSize - 1) / 2; l++) {
						for (int m = j - (winSize - 1) / 2; m <= j + (winSize - 1) / 2; m++) {
							if (l < 0 || m < 0 || m + k >= w || m >= w || l >= h) {
								continue;
							}
							else {
								Srl += IR[l * w + m] * IL[l * w + m + k];
								Sr += IR[l * w + m];
								Sl += IL[l * w + m + k];
								Srr += IR[l * w + m ] * IR[l * w + m];
								Sll += IL[l * w + m + k] * IL[l * w + m + k];
							}
						}
					}
					sum = (Srl - Sr * Sl / (winSize * winSize)) / sqrt((Srr - Sr * Sr / (winSize *
					winSize)) * (Sll - Sl * Sl / (winSize * winSize)));
				}
				if (sum > max) {
					max = sum;
					pos = k;
				}
			}
			res[i * w + j] = pos;
		}
	}
	return res;
}
/*�������ܣ����㴰���е����������������Ȩ��ϵ����
���룺������������ֱ���CIE Labɫ�ʿռ��µ�L��a��b��ֵ
���������Ȩ��ϵ��
*/
double weight(double IL0, double IL1, double IL2, double IR0, double IR1, double IR2, double Gab) {
	double cpq = sqrt((IL0 - IR0) * (IL0 - IR0) + (IL1 - IR1) * (IL1 - IR1) + (IL2 - IR2) * (IL2 - IR2));
	double res = K * exp(-(cpq / rc + Gab / rp));
	return res;
}
/**��������:���ض�Ӧ���ص�ɫ��ǿ�ȵĲ�ľ���ֵ
���룺������Ӧ���صĻҶ�ֵ
���������ֵ�Ĳ�ľ���ֵ
*/
double e0(double il0, double il1) {
	double res = 0;
	res = abs(il0 - il1);
	return res;
}
/**�������ܣ��������������ͼ���д�����ASW�ķ�ʽ�õ��������Ӳ�ͼ��
���룺IL��IR ������ͼ
h, w ͼƬ�ĳ���
flag ��ǣ���������������Ӳ�ͼ�������Ӳ�ͼ
��������ش���������������洢���Ӳ�ͼ
*/
int* ASW(Mat IL, Mat IR, int h, int w, char flag) {
	int* res = new int[h * w];
	Mat ILCIE(h, w, CV_8UC3, Scalar(0, 0, 0)), IRCIE(h, w, CV_8UC3, Scalar(0, 0, 0));
	cvtColor(IL, ILCIE, CV_BGR2Lab);
	cvtColor(IR, IRCIE, CV_BGR2Lab);
	cvtColor(IR, IR, CV_BGR2GRAY);
	cvtColor(IL, IL, CV_BGR2GRAY);
	double* il0 = new double[h * w], *ir0 = new double[h * w];
	double* ILCIE0 = new double[h * w], *ILCIE1 = new double[h * w], *ILCIE2 = new double[h * w], 
		*IRCIE0 = new double[h * w], *IRCIE1 = new double[h * w], *IRCIE2 = new double[h * w];
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			Scalar tmp = IL.at<uchar>(i, j);
			il0[i * w + j] = tmp.val[0];
			tmp = IR.at<uchar>(i, j);
			ir0[i * w + j] = tmp.val[0];
			tmp = ILCIE.at<uchar>(i, j);
			ILCIE0[i * w + j] = tmp.val[0];
			ILCIE1[i * w + j] = tmp.val[1];
			ILCIE2[i * w + j] = tmp.val[2];
			tmp = IRCIE.at<uchar>(i, j);
			IRCIE0[i * w + j] = tmp.val[0];
			IRCIE1[i * w + j] = tmp.val[1];
			IRCIE2[i * w + j] = tmp.val[2];
		}
	}
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int pos = 0;
			double min = 100000000000000.0;
			for (int k = 0; k <= dmax; k++) {
				double sum1 = 0;
				double sum2 = 0;
				double sum = 0;
				double tmp = 0;
				if (flag == 'l') {
					for (int l = i - (winSize - 1) / 2; l <= i + (winSize - 1) / 2; l++) {
						for (int m = j - (winSize - 1) / 2; m <= j + (winSize - 1) / 2; m++) {
							if (m < 0 || l < 0 || m - k < 0 || m >= w || l >= h || j - k < 0) {
								continue;
							}
							else {
								tmp = weight(ILCIE0[i * w + j], ILCIE1[i * w + j], ILCIE2[i * w + j], ILCIE0[l * w + m], ILCIE1[l * w + m], ILCIE2[l * w + m],
									sqrt((i - l) * (i - l) + (j - m) * (j - m))) * weight(IRCIE0[i * w + j - k], IRCIE1[i * w + j - k], IRCIE2[i * w + j - k],
									IRCIE0[l * w + m - k], IRCIE1[l * w + m - k], IRCIE2[l * w + m - k], sqrt((i - l) * (i - l) + (j - m) * (j - m)));
								sum1 += tmp;
								sum2 += tmp * e0(il0[l * w + m], ir0[l * w + m - k]);
							}
						}
					}
					sum = sum2 / sum1;
				}
				else {
					for (int l = i - (winSize - 1) / 2; l <= i + (winSize - 1) / 2; l++) {
						for (int m = j - (winSize - 1) / 2; m <= j + (winSize - 1) / 2; m++) {
							if (m < 0 || l < 0 || m + k >= w || m >= w || l >= h || j + k >= w) {
								continue;
							}
							else {
								tmp = weight(ILCIE0[i * w + j + k], ILCIE1[i * w + j + k], ILCIE2[i * w + j + k], ILCIE0[l * w + m + k], ILCIE1[l * w + m + k], ILCIE2[l * w + m + k],
									sqrt((i - l) * (i - l) + (j - m) * (j - m))) * weight(IRCIE0[i * w + j], IRCIE1[i * w + j], IRCIE2[i * w + j],
									IRCIE0[l * w + m], IRCIE1[l * w + m], IRCIE2[l * w + m], sqrt((i - l) * (i - l) + (j - m) * (j - m)));
								sum1 += tmp;
								sum2 += tmp * e0(il0[l * w + m + k], ir0[l * w + m]);
							}
						}
					}
					sum = sum2 / sum1;
				}
				if (sum < min) {
					min = sum;
					pos = k;
				}
			}
			res[i * w + j] = pos;
		}
	}
	return res;
}
/**��������:������������SSD��NCC��ASW����ȥ���в�ͬ���������ԡ���ͬ������ͨ���滻�ļ����е��ַ�����ʵ�֡�
*/
int main() {
	//����������ͼ
	Mat IL = imread("./test/Monopoly/view1.png");
    Mat IR = imread("./test/Monopoly/view5.png");
	int h = IL.rows;
	int w = IL.cols;
	Mat l(h, w, CV_8UC1, Scalar(0, 0, 0)), r(h, w, CV_8UC1, Scalar(0, 0, 0));
	Mat DLSSD(h, w, CV_8UC1, Scalar(0, 0, 0)), DRSSD(h, w, CV_8UC1, Scalar(0, 0, 0));
	Mat DLNCC(h, w, CV_8UC1, Scalar(0, 0, 0)), DRNCC(h, w, CV_8UC1, Scalar(0, 0, 0));
	Mat DLASW(h, w, CV_8UC1, Scalar(0, 0, 0)), DRASW(h, w, CV_8UC1, Scalar(0, 0, 0));
	//����ɫͼת���ɻҶ�ͼ
	cvtColor(IL, l, CV_BGR2GRAY);
	cvtColor(IR, r, CV_BGR2GRAY);
	//���Ҷ�ͼת����˫�����������ʽ
	double* il = new double[h*w];
	double* ir = new double[h*w];
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			Scalar a = l.at<uchar>(i, j);
			il[i * w + j] = a.val[0];
			a = r.at<uchar>(i, j);
			ir[i * w + j] = a.val[0] + 10;
		}
	}
	//�����ַ�ʽ�����Ӳ�ͼ
	DLASW = scale(ASW(IL, IR, h, w, 'l'), h, w);
	DRASW = scale(ASW(IL, IR, h, w, 'r'), h, w);
	DLSSD = scale(SSD(il, ir, h, w, 'l'), h, w);
	DRSSD = scale(SSD(il, ir, h, w, 'r'), h, w);
	DLNCC = scale(NCC(il, ir, h, w, 'l'), h, w);
	DRNCC = scale(NCC(il, ir, h, w, 'r'), h, w);
	//�������Ӳ�ͼ
	Mat GDL = imread("./test/Monopoly/disp1.png");
	Mat GDR = imread("./test/Monopoly/disp5.png");
	//���Աȣ��������Ӳ�ͼ
	cout << "Left ASW:" << evaluator(DLASW, GDL) << endl;
	cout << "Right ASW:" << evaluator(DRASW, GDR) << endl;
	imwrite("./res/Monopoly_disp1_ASW.png", DLASW);
	imwrite("./res/Monopoly_disp5_ASW.png", DRASW);

	cout << "Left SSD:" << evaluator(DLSSD, GDL) << endl;
	cout << "Right SSD:" << evaluator(DRSSD, GDR) << endl;
	imwrite("./res/Monopoly_disp1_SSD.png", DLSSD);
	imwrite("./res/Monopoly_disp5_SSD.png", DRSSD);

	cout << "Left NCC:" << evaluator(DLNCC, GDL) << endl;
	cout << "Right NCC:" << evaluator(DRNCC, GDR) << endl;
	imwrite("./res/Monopoly_disp1_NCC.png", DLNCC);
	imwrite("./res/Monopoly_disp5_NCC.png", DRNCC);
	waitKey(10000);
	system("pause");
	return 0;
}