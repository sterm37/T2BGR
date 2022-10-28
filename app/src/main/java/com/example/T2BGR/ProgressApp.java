package com.example.T2BGR;

import android.app.Activity;
import android.app.Application;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.text.TextUtils;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatDialog;

public class ProgressApp extends Application {

    private static ProgressApp progressApp;
    AppCompatDialog progressDialog;

    public static ProgressApp getInstance(){
        return progressApp;
    }

    public void onCreate(){
        super.onCreate();
        progressApp = this;
    }

    public void progressON(Activity activity, String msg)
    {
        Toast.makeText(getApplicationContext(),"dialog on3", Toast.LENGTH_SHORT).show();

        if (activity == null || activity.isFinishing()){
            return;
        }

        if(progressDialog != null && progressDialog.isShowing()){
            progressSET(msg);
        }else{
            progressDialog = new AppCompatDialog(activity);
            progressDialog.setCancelable(false);
            progressDialog.getWindow().setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
            progressDialog.setContentView(R.layout.progress_dialog);
            progressDialog.show();
        }

        TextView progressMSG = (TextView) progressDialog.findViewById(R.id.ProgressMessage);
        if(!TextUtils.isEmpty(msg)){
            progressMSG.setText(msg);
        }

    }

    public void progressSET(String msg)
    {
        if(progressDialog == null || !progressDialog.isShowing()){
            return;
        }

        TextView progressMSG = (TextView) progressDialog.findViewById(R.id.ProgressMessage);
        if(!TextUtils.isEmpty(msg)){
            progressMSG.setText(msg);
        }

    }

    public void progressOFF()
    {
        if(progressDialog != null && progressDialog.isShowing()){
            progressDialog.dismiss();
        }
    }

}
