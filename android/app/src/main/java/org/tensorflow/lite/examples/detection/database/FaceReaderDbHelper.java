package org.tensorflow.lite.examples.detection.database;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

public class FaceReaderDbHelper extends SQLiteOpenHelper {
    // If you change the database schema, you must increment the database version.
    public static final int DATABASE_VERSION = 1;
    public static final String DATABASE_NAME = "FaceReader.db";

    private static final String SQL_CREATE_ENTRIES =
            "CREATE TABLE " + FaceReaderContract.FaceEntry.TABLE_NAME + " (" +
                    FaceReaderContract.FaceEntry._ID + " INTEGER PRIMARY KEY," +
                    FaceReaderContract.FaceEntry.COLUMN_NAME_FACE + " TEXT," +
                    FaceReaderContract.FaceEntry.COLUMN_NAME_NAME + " TEXT)";

    private static final String SQL_DELETE_ENTRIES =
            "DROP TABLE IF EXISTS " + FaceReaderContract.FaceEntry.TABLE_NAME;

    public FaceReaderDbHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }
    public void onCreate(SQLiteDatabase db) {
        db.execSQL(SQL_CREATE_ENTRIES);
    }
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        // This database is only a cache for online data, so its upgrade policy is
        // to simply to discard the data and start over
        db.execSQL(SQL_DELETE_ENTRIES);
        onCreate(db);
    }
    public void onDowngrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        onUpgrade(db, oldVersion, newVersion);
    }
}