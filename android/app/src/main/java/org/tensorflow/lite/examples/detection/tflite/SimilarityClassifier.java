/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.graphics.Bitmap;
import android.graphics.RectF;
import android.text.TextUtils;

import java.io.ByteArrayOutputStream;
import java.util.Collections;
import java.util.List;

/** Generic interface for interacting with different recognition engines. */
public interface SimilarityClassifier {

  void register(String name, Recognition recognition);

  List<Recognition> recognizeImage(Bitmap bitmap, boolean getExtra);

  void enableStatLogging(final boolean debug);

  String getStatString();

  void close();

  void setNumThreads(int num_threads);

  void setUseNNAPI(boolean isChecked);

  /** An immutable result returned by a Classifier describing what was recognized. */
  public class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /** Display name for the recognition. */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Lower should be better.
     */
    private final Float distance;
    private Object extra;

    /** Optional location within the source image for the location of the recognized object. */
    private RectF location;
    private Integer color;
    private Bitmap crop;

    public Recognition(
            final String id, final String title, final Float distance, final RectF location) {
      this.id = id;
      this.title = title;
      this.distance = distance;
      this.location = location;
      this.color = null;
      this.extra = null;
      this.crop = null;
    }

    public Recognition(final String rec) {
      this.id = getValue(rec, "id: ");
      this.title = getValue(rec, "title: ");
      this.distance = Float.parseFloat(getValue(rec, "distanceRaw: "));
      String locationRaw = getValue(rec, "RectF(");
      String[] locationRawParsed = locationRaw.substring(0, locationRaw.length() - 1).split(",");
      float left = Float.parseFloat(locationRawParsed[0]);
      float top = Float.parseFloat(locationRawParsed[1]);
      float right = Float.parseFloat(locationRawParsed[2]);
      float bottom = Float.parseFloat(locationRawParsed[3]);

      this.location = new RectF(left, top, right, bottom);
      String extraP = getValue(rec,"extra: ");
      if(!extraP.equals("")) {
        String[] values = extraP.split(",");
        float[][] array = new float[1][values.length];
        for (int i = 0; i < values.length; i++) {
          array[0][i] = Float.parseFloat(values[i]);
        }
        this.extra = array;
      } else {
        this.extra = null;
      }
      this.color = null;
      this.crop = null;
    }

    private String getValue(final String rec, String prefix) {
      final int prefixIndex = rec.indexOf(prefix);
      StringBuilder result = new StringBuilder();
      for(int i = prefixIndex + prefix.length(); i < rec.length(); i++) {
        if(rec.charAt(i) == ';') {
          return result.toString();
        }
        result.append(rec.charAt(i));
      }
      return result.toString();
    }

    public void setExtra(Object extra) {
        this.extra = extra;
    }
    public Object getExtra() {
        return this.extra;
    }

    public void setColor(Integer color) {
       this.color = color;
    }

    public String getId() {
      return id;
    }

    public String getTitle() {
      return title;
    }

    public Float getDistance() {
      return distance;
    }

    public RectF getLocation() {
      return new RectF(location);
    }

    public void setLocation(RectF location) {
      this.location = location;
    }

    @Override
    public String toString() {
      String resultString = "";
      if (id != null) {
        resultString += "id: " + id + "; ";
      }

      if (title != null) {
        resultString += "title: " + title + "; ";
      }

      if (distance != null) {
        resultString += "distance: " + String.format("(%.1f%%) ", distance * 100.0f) + "; ";
        resultString += "distanceRaw: " + distance + "; ";
      }

      if (location != null) {
        resultString += "location: " + location + "; ";
      }

      if (extra != null) {
        StringBuilder extraJoined = new StringBuilder();
        for (float f:((float[][]) extra)[0]) {
          extraJoined.append(f).append(",");
        }
        resultString += "extra: " + extraJoined.toString() + "; ";
      }

      return resultString.trim();
    }

    public Integer getColor() {
      return this.color;
    }

    public void setCrop(Bitmap crop) {
      this.crop = crop;
    }

    public Bitmap getCrop() {
      return this.crop;
    }
  }
}
