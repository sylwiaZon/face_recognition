package org.tensorflow.lite.examples.detection.database;

import android.provider.BaseColumns;

public class FaceReaderContract {
    private FaceReaderContract() {}

    /* Inner class that defines the table contents */
    public static class FaceEntry implements BaseColumns {
        public static final String TABLE_NAME = "faces";
        public static final String COLUMN_NAME_FACE = "face";
        public static final String COLUMN_NAME_NAME = "name";
    }
}
