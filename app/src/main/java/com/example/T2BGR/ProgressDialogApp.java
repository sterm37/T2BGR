package com.example.T2BGR;

import android.app.Dialog;
import android.content.Context;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.util.Log;
import android.view.Window;
import android.widget.TextView;

import androidx.annotation.NonNull;

public class ProgressDialogApp extends Dialog
{

    TextView progressMSG;

    public ProgressDialogApp(@NonNull Context context)
    {
        super(context);
        Log.d("Click", "dialog clicked2");
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.progress_dialog);
        Log.d("Click", "dialog clicked2");
        getWindow().setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
        setCancelable(false);
        progressMSG = findViewById(R.id.ProgressMessage);

    }


}
