package com.github.levyfan.reid.codebook;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Floats;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.opencv_core;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * @author fanliwen
 */
public class CodeBook {

    public List<double[]> codebook(Iterable<double[]> feature, int size) {
        List<float[]> floatList = new ArrayList<>();
        for (double[] doubles : feature) {
            floatList.add(Floats.toArray(Doubles.asList(doubles)));
        }
        float[] floats = Floats.concat(floatList.toArray(new float[0][]));

        opencv_core.Mat points = new opencv_core.Mat(
                floatList.size(),
                floatList.get(0).length,
                opencv_core.CV_32F,
                new FloatPointer(floats));

        opencv_core.Mat labels = new opencv_core.Mat();
        opencv_core.Mat centers = new opencv_core.Mat();

        opencv_core.kmeans(
                points,
                size,
                labels,
                new opencv_core.TermCriteria(opencv_core.TermCriteria.MAX_ITER, 100, 0),
                1,
                opencv_core.KMEANS_PP_CENTERS,
                centers);

        List<double[]> doubles = new ArrayList<>();
        for (int i = 0; i < centers.rows(); i++) {
            opencv_core.Mat row = centers.row(i);
            FloatBuffer buffer = row.createBuffer();

            float[] floatArray = new float[floatList.get(0).length];
            buffer.get(floatArray);

            doubles.add(Doubles.toArray(Floats.asList(floatArray)));
        }
        return doubles;
    }
}
