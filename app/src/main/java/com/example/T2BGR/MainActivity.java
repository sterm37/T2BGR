package com.example.T2BGR;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.FileProvider;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.UUID;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;


public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    private  Handler mHandler;
    private Socket socket;
    private DataOutputStream dos;
    private DataInputStream dis;
    private static String serIP = "192.168.0.10";
    //private static String serIP = "10.0.2.2";
    private static int serPort = 22334;
    private String img_path;

    final String TAG = getClass().getSimpleName();
    ImageButton cameraBtn;
    Button sendBtn;
    Button AddBtn;
    EditText sndMsg;
    ImageView geneImage;
    ImageView captImage;
    ProgressDialogApp progressApp;
    Uri photoUri;
    private Bitmap img;
    private Bitmap rotatedBitmap = null;
    ImageView photoImageView;
    private  String mImage;

    static String mark = null;
    static String shape = null;
    private final Charset UTF8_CHARSET = Charset.forName("UTF-8");

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // 부모 클래스에 있는 onCreate() 함수를 호출
        setContentView(R.layout.activity_main);

        cameraBtn = findViewById(R.id.cameraButton);
        cameraBtn.setOnClickListener(this);

        sendBtn = findViewById(R.id.SendButton);
        sendBtn.setOnClickListener(this);
        AddBtn = findViewById(R.id.AddButton);
        AddBtn.setOnClickListener(this);

        sndMsg = findViewById(R.id.imageText);
        geneImage = findViewById(R.id.geneImage);
        captImage = findViewById(R.id.captImage);

        progressApp = new ProgressDialogApp(this);

        if(checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
                && checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED){
        }else{
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }

    }

    private String dateName(long dateTaken){
        Date date = new Date(dateTaken);
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH.mm.ss");
        return dateFormat.format(date);
    }

    private void savePicture(Bitmap imgBit){
        UUID uuid = UUID.randomUUID();
        String date = "IMG_" + dateName(System.currentTimeMillis()) + uuid.toString();
        File dir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),date);
        FileOutputStream fos;
        try{
            fos = new FileOutputStream( dir.toString()+".jpg");
            imgBit.compress(Bitmap.CompressFormat.JPEG, 100, fos);
            Log.d("SAVE", "SAVE COMPLETE");
            fos.flush();
            fos.close();
        }catch (Exception e){
            e.printStackTrace();
            Log.w("SAVE", "SAVE Failed");
        }
    }

    public String readString (DataInputStream dis) throws IOException{
        int length = dis.readInt();
        byte[] data = new byte[length];
        dis.readFully(data, 0, length);
        String text = new String(data, StandardCharsets.UTF_8);

        return text;
    }

    public byte[] InStreamToByteArr(int dataLen, DataInputStream dis){
        int loop = (int)(dataLen/1024);
        byte[] resByte = new byte[dataLen];
        int offset = 0;
        try {
            for(int i = 0; i < loop; i++){
                dis.readFully(resByte, offset, 1024);
                offset += 1024;
            }
            dis.readFully(resByte, offset, dataLen-(loop*1024));
        }catch (IOException e){
            e.printStackTrace();
        }

        return resByte;
    }

    void connect(String msg){
        //버튼을 눌르면 함수를 실행하게 만듬
        String exitMsg;
        exitMsg = "/exit";
        mHandler = new Handler();
        //쓰레드 생성
        Log.d("connect", "Thread Gene...");
        Thread sendText = new Thread() {
            int dataLen = 0;
            byte[] img = null;

            @Override
            public void run() {
                try {
                    Log.d("connect", "connecting...");
                    socket = new Socket(serIP, serPort);
                    Log.d("connect","connected!");
                }catch (IOException e1){
                    Log.w("connect", "Not Connect");
                    e1.printStackTrace();
                }

                try {
                    Log.d(": ", "Android to Server Sending");
                    dis = new DataInputStream(socket.getInputStream());
                    dos = new DataOutputStream(socket.getOutputStream());
                    Log.d("Buffer", "Buffer generated");
                }catch (IOException e2){
                    e2.printStackTrace();
                    Log.w("Buffer", "Not Buffer gen");
                }

                try {

                    dos.writeUTF(msg);
                    dos.flush();

                    ByteArrayOutputStream byteArray = new ByteArrayOutputStream();
                    dataLen = dis.readInt();
                    img = InStreamToByteArr(dataLen, dis);

                    dos.writeUTF(exitMsg);
                    dos.flush();

                    if(img == null){
                        Log.w("dataNull", "imgData NULL FUCK");
                    }else{
                        Log.d("dataIn", "imgData IN FUCK");
                    }

                    Bitmap imgBitmap = null;
                    imgBitmap = BitmapFactory.decodeByteArray(img, 0, img.length);

                    savePicture(imgBitmap);

                    if(imgBitmap == null){
                        Log.w("dataNull", "imgBitmap NULL FUCK");
                    }else{
                        Log.d("dataIn", "imgBitmap IN FUCK");
                    }

                    imgBitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArray);

                    geneImage.setImageBitmap(imgBitmap);

                }catch (Exception e3){
                    Log.w("error", "error occur");
                    e3.printStackTrace();
                }
            }
        };

        sendText.start();
        try {
            sendText.join();
        }catch (InterruptedException e){

        }


    }

    public void progressON(){
        if(progressApp != null && progressApp.isShowing()){
            progressApp.show();
        }else{
            Log.d("Click", "dialog clicked1");
            progressApp.show();
            Log.d("Click", "dialog clicked3");
        }
    }

    public void progressON(String msg){
        Toast.makeText(getApplicationContext(),"dialog on2", Toast.LENGTH_SHORT).show();

        if(progressApp != null && progressApp.isShowing()){

        }else{
            progressApp.show();
        }
    }

    public void progressSET(String msg)
    {

    }

    public void progressOFF(){
        if(progressApp != null && progressApp.isShowing()){
            progressApp.dismiss();
        }
    }

    private void progressFunc() {

        progressON("Loading...");

        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                progressOFF();
            }
        }, 2000);

    }

    public void onClick(View v){

        switch (v.getId()){
            case R.id.cameraButton:
                Toast.makeText(getApplicationContext(),"cameraButtonClicked", Toast.LENGTH_SHORT).show();
                dispatchTakePictureIntent();
                Log.d("Click", "CameraButton clicked");
                break;
            case R.id.SendButton:
                //Toast.makeText(getApplicationContext(),"sendButtonClicked", Toast.LENGTH_SHORT).show();
                String msg;
                msg = sndMsg.getText().toString();

                Toast.makeText(getApplicationContext(),"dialog on", Toast.LENGTH_SHORT).show();

                Log.d("Click", "dialog1");
                progressFunc();
                connect(msg);
                //서버접속 끝
                //progressOFF();

                sndMsg.setText("");

                Toast.makeText(getApplicationContext(),"dialog off", Toast.LENGTH_SHORT).show();
                break;
            case R.id.AddButton:

                if(geneImage.getDrawable() == null || captImage.getDrawable() == null){
                    Toast.makeText(getApplicationContext(),"두 이미지가 준비되어 있지 않습니다.", Toast.LENGTH_SHORT).show();
                }else{
                    BitmapDrawable imgDraw1 = (BitmapDrawable) geneImage.getDrawable();
                    BitmapDrawable imgDraw2 = (BitmapDrawable) captImage.getDrawable();
                    Bitmap img1 = imgDraw1.getBitmap();
                    Bitmap img2 = imgDraw2.getBitmap();
                    Bitmap dst = ProcImg(img1, img2);
                    savePicture(dst);
                }
                break;
        }
    }



    private File createImageFile() throws IOException{
        String imageFileName = "IMG_" + dateName(System.currentTimeMillis());
        File storageDir =
                getExternalFilesDir(Environment.getExternalStoragePublicDirectory
                        (Environment.DIRECTORY_PICTURES).toString());

        File image = File.createTempFile(
                imageFileName,
                ".jpg",
                storageDir
        );

        return image;

    }

    protected void onActivityResult(int requestCode, int resultCode, Intent intent){
        super.onActivityResult(requestCode, resultCode, intent);

        if(requestCode == 0 && resultCode == RESULT_OK){
            //captImage.setImageURI(photoUri);
            //BitmapDrawable imgDraw2 = (BitmapDrawable) captImage.getDrawable();
            //Bitmap imgBM = imgDraw2.getBitmap();
            Bitmap imgBM = getImgBM2("human");
            captImage.setImageBitmap(imgBM);
            GrabCutAlgo(imgBM);
            //savePicture(imgBM);

        }

    }

    public void SleepProc(){
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                Log.d("SLEEP", "Sleep 0.5s");
            }
        }, 1000);
    }

    //TEST 사진 가져오기
    public Bitmap getImgBM(){
        Bitmap bm = null;

        File imgDir = null;
        try {
            imgDir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),"/human.jpg");
            Log.d("utb", imgDir.toString());
        }catch (Exception e){
            e.printStackTrace();
        }

        try{
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            bm = BitmapFactory.decodeStream(new FileInputStream(imgDir), null, options);
        }catch (Exception e){
            e.printStackTrace();
        }

        return bm;
    }

    public Bitmap getImgBM2(String str1){
        Bitmap bm1 = null;

        String fName = "/" + str1 + ".jpg";

        File imgDir = null;
        try {
            imgDir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),fName);
            Log.d("utb", imgDir.toString());
        }catch (Exception e){
            e.printStackTrace();
        }

        try{
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            bm1 = BitmapFactory.decodeStream(new FileInputStream(imgDir), null, options);
        }catch (Exception e){
            e.printStackTrace();
        }

        return bm1;
    }

    private void dispatchTakePictureIntent(){
        OpenCVLoader.initDebug();
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        Log.d("photo", "open camera");

        File photoFile = null;
        try{
            photoFile = createImageFile();
        }catch (IOException e){
            e.printStackTrace();
        }

        if(photoFile != null){
            photoUri = FileProvider.getUriForFile(this,
                    getPackageName() + ".provider",
                    photoFile);
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
            startActivityForResult(takePictureIntent, 0);
        }

    }

    public Bitmap UriToBitmap(Uri uri){
        Bitmap bm = null;
        try{
            bm = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
            Log.d("utb", "1");
        }catch (FileNotFoundException e){
            e.printStackTrace();
            Log.w("utb", "no File");
        }catch (IOException e){
            e.printStackTrace();
            Log.w("utb", "error");
        }
        return bm;
    }

    public void GrabCutAlgo(Bitmap bitmap){
        Bitmap imgBit = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Point tl = new Point();
        Point br = new Point();

        Mat img = new Mat();
        Utils.bitmapToMat(imgBit, img);
        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGBA2RGB);

        int r = img.rows();
        int c = img.cols();
        Point p1 = new Point(c / 100, r / 100);
        Point p2 = new Point(c - c / 100, r - r / 100);
        Rect rect = new Rect(p1, p2);

        Mat background = new Mat(img.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
        Mat firstMask = new Mat();
        Mat bgModel = new Mat();
        Mat fgModel = new Mat();
        Mat mask;
        Mat source = new Mat(1, 1, CvType.CV_8U, new Scalar(Imgproc.GC_PR_FGD));
        Mat dst = new Mat();

        Log.d("grabCut", "start");
        Imgproc.grabCut(img, firstMask, rect, bgModel, fgModel, 5, Imgproc.GC_INIT_WITH_RECT);
        Core.compare(firstMask, source, firstMask, Core.CMP_EQ);
        Log.d("grabCut", "end");

        Mat foreground = new Mat(img.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));

        img.copyTo(foreground, firstMask);

        Scalar color = new Scalar(255, 0, 0, 255);
        Imgproc.rectangle(img, tl, br, color);

        Mat tmp = new Mat();
        Imgproc.resize(background, tmp, img.size());
        background = tmp;
        mask = new Mat(foreground.size(), CvType.CV_8UC1,
                new Scalar(255, 255, 255));

        Imgproc.cvtColor(foreground, mask, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(mask, mask, 254, 255, Imgproc.THRESH_BINARY_INV);
        System.out.println();
        Mat vals = new Mat(1, 1, CvType.CV_8UC3, new Scalar(0.0));
        background.copyTo(dst);

        background.setTo(vals, mask);

        Core.add(background, foreground, dst, mask);
        Bitmap grabCutImage = Bitmap.createBitmap(dst.cols(), dst.rows(), Bitmap.Config.ARGB_8888);
        Bitmap processedImage = Bitmap.createBitmap(dst.cols(), dst.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(dst, grabCutImage);

        captImage.setImageBitmap(grabCutImage);
        firstMask.release();
        source.release();
        bgModel.release();
        fgModel.release();

    }

    public Bitmap ProcImg(Bitmap img1, Bitmap img2){
        OpenCVLoader.initDebug();
        Bitmap img1Bit = img1.copy(Bitmap.Config.ARGB_8888, true);
        Bitmap img2Bit = img2.copy(Bitmap.Config.ARGB_8888, true);//해당 비트맵형태로 복사
        Log.d("ADD", "0");

        Mat img_bg = new Mat();//배경
        Mat img_fg = new Mat();//사진
        Mat img_bgG = new Mat();//회색 배경
        Mat img_fgG = new Mat();//회색 사진
        Mat mask = new Mat();
        Mat mask_inv = new Mat();

        Utils.bitmapToMat(img1Bit, img_bg);
        Utils.bitmapToMat(img2Bit, img_fg);//비트맵 데이터를 행렬 타입으로 변환

        Log.d("ADD", "0-1");
        Imgproc.cvtColor(img_bg, img_bg, Imgproc.COLOR_RGBA2RGB);
        Imgproc.cvtColor(img_fg, img_fg, Imgproc.COLOR_RGBA2RGB);

        int fgW = img_fg.width();
        int fgH = img_fg.height();
        Log.d("ADD", String.valueOf(fgW));
        Log.d("ADD", String.valueOf(fgH));
        if(fgW > 960 || fgH > 540){
            fgW = (int) (fgW / Math.ceil(fgW / 960.0));
            fgH = (int) (fgH / Math.ceil(fgH / 540.0));
        }
        Log.d("ADD", "1 start");
        Log.d("ADD", String.valueOf(fgW));
        Log.d("ADD", String.valueOf(fgH));

        Size bgSize = new Size(1920, 1080);
        Size fgSize = new Size(fgW, fgH);

        Log.d("ADD", "2");

        Imgproc.resize(img_bg, img_bg, bgSize, 0, 0, Imgproc.INTER_LINEAR);
        Imgproc.resize(img_fg, img_fg, fgSize, 0, 0, Imgproc.INTER_AREA);
        Imgproc.cvtColor(img_bg, img_bgG, Imgproc.COLOR_RGB2GRAY);
        Imgproc.cvtColor(img_fg, img_fgG, Imgproc.COLOR_RGB2GRAY);
        //Bitmap temp3 = Bitmap.createBitmap(img_fgG.cols(), img_fgG.rows(), Bitmap.Config.ARGB_8888);
        //Utils.matToBitmap(img_fgG, temp3);
        //savePicture(temp3);

        Log.d("ADD", "3-2");

        int x = (int)img_bg.size().width / 2 - fgW / 2 ;
        int y = (int)img_bg.size().height / 2;
        Log.d("bg x", String.valueOf(img_bg.size()));
        Rect rect = new Rect(x, y, fgW, fgH);

        Log.d("roi x", String.valueOf(x));
        Log.d("roi y", String.valueOf(y));

        Log.d("ADD", "3-3");
        Log.d("roi size", String.valueOf(rect.size()));

        Log.d("ADD", "3-4");
        Mat roi = img_bg.submat(rect);

        Log.d("ADD", "4");
        //Bitmap temp = Bitmap.createBitmap(roi.cols(), roi.rows(), Bitmap.Config.ARGB_8888);
        //Utils.matToBitmap(roi, temp);
        //savePicture(temp);
        //Log.d("ADD", String.valueOf(img_bg.channels()));

        Imgproc.threshold(img_fgG, mask, 254, 255, Imgproc.THRESH_BINARY);
        Core.bitwise_not(mask, mask_inv);

        Imgproc.cvtColor(mask, mask, Imgproc.COLOR_GRAY2RGB);
        Imgproc.cvtColor(mask_inv, mask_inv, Imgproc.COLOR_GRAY2RGB);
        //Bitmap temp5 = Bitmap.createBitmap(mask.cols(), mask.rows(), Bitmap.Config.ARGB_8888);
        //Utils.matToBitmap(mask, temp5);
        //savePicture(temp5);
        //Bitmap temp6 = Bitmap.createBitmap(mask_inv.cols(), mask_inv.rows(), Bitmap.Config.ARGB_8888);
        //Utils.matToBitmap(mask_inv, temp6);
        //savePicture(temp6);


        Log.d("ADD", "5");

        Core.bitwise_and(roi, mask, roi);
        //Bitmap temp4 = Bitmap.createBitmap(roi.cols(), roi.rows(), Bitmap.Config.ARGB_8888);
        //Utils.matToBitmap(roi, temp4);
        //savePicture(temp4);


        Core.bitwise_and(img_fg, mask_inv, img_fg);
        //Bitmap temp2 = Bitmap.createBitmap(img_fg.cols(), img_fg.rows(), Bitmap.Config.ARGB_8888);
        //Utils.matToBitmap(img_fg, temp2);
        //savePicture(temp2);

        Log.d("ADD", "6");

        Core.bitwise_or(roi, img_fg, roi);
        //Bitmap temp = Bitmap.createBitmap(roi.cols(), roi.rows(), Bitmap.Config.ARGB_8888);
        //Utils.matToBitmap(roi, temp);
        //savePicture(temp);

        Log.d("ADD", "7");

        Bitmap dst = Bitmap.createBitmap(img_bg.cols(), img_bg.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img_bg, dst);

        captImage.setImageBitmap(dst);
        //savePicture(dst);

        return dst;
    }

}
