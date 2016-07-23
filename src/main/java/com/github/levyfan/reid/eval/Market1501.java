package com.github.levyfan.reid.eval;

import com.google.common.collect.TreeMultimap;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.util.Pair;

import java.util.*;

/**
 * @author fanliwen
 */
public class Market1501 {

    private static Pair<Double, double[]> compute(int[] index, Set<Integer> goodImage, Set<Integer> junkImage) {
        double[] cmc = new double[index.length];
        double ap = 0;

        double old_recall = 0;
        double old_precision = 1.0;

        int intersect_size = 0;
        int j = 0;
        int good_now = 0;
        int njunk = 0;

        for (int n = 0; n < index.length; n++) {
            int flag = 0;
            if (goodImage.contains(index[n])) {
                Arrays.fill(cmc, n - njunk, cmc.length, 1);
                flag = 1;
                good_now ++;
            }
            if (junkImage.contains(index[n])) {
                njunk ++;
                continue;
            }

            if (flag == 1) {
                intersect_size ++;
            }
            j ++;
            double recall = ((double) intersect_size) / ((double) goodImage.size());
            double precision = ((double) intersect_size) / ((double) j);
            ap += (recall - old_recall) * (precision + old_precision) / 2;

            old_recall = recall;
            old_precision = precision;

            if (good_now == goodImage.size()) {
                return Pair.create(ap, cmc);
            }
        }
        return Pair.create(ap, cmc);
    }

    public Pair<Double, double[]> eval(
            List<double[]> queryHist,
            List<Pair<Integer, Integer>> queryIdAndCam,
            List<double[]> testHist,
            List<Pair<Integer, Integer>> testIdAndCam) {
        double[] ap = new double[queryHist.size()];
        RealMatrix cmc = MatrixUtils.createRealMatrix(queryHist.size(), testHist.size());

        for (int k = 0; k < queryHist.size(); k ++) {
            ArrayRealVector query = new ArrayRealVector(queryHist.get(k));

            // sort by score
            int[] index = new int[testHist.size()];
            TreeMultimap<Double, Integer> multimap = TreeMultimap.create();
            for (int i = 0; i < testHist.size(); i++) {
                multimap.put(query.dotProduct(new ArrayRealVector(testHist.get(i))), i);
            }
            Iterator<Map.Entry<Double, Integer>> iterator = multimap.entries().iterator();
            for (int i = 0; i < testHist.size(); i++) {
                index[i] = iterator.next().getValue();
            }

            int queryId = queryIdAndCam.get(k).getFirst();
            int queryCam = queryIdAndCam.get(k).getSecond();
            Set<Integer> goodIndex = new HashSet<>();
            Set<Integer> junkIndex = new HashSet<>();
            for (int i = 0; i < testHist.size(); i++) {
                int testId = testIdAndCam.get(i).getFirst();
                int testCam = testIdAndCam.get(i).getSecond();
                if (testId == queryId && testCam != queryCam) {
                    goodIndex.add(i);
                }
                if (testId == queryId && testCam == queryCam) {
                    junkIndex.add(i);
                }
                if (testId == -1) {
                    junkIndex.add(i);
                }
            }

            Pair<Double, double[]> pair = compute(index, goodIndex, junkIndex);
            ap[k] = pair.getFirst();
            cmc.setRow(k, pair.getSecond());
        }

        double[] cmcArray = new double[cmc.getColumnDimension()];
        for (int col = 0; col < cmc.getColumnDimension(); col ++) {
            cmcArray[col] = new Mean().evaluate(cmc.getColumn(col));
        }
        return Pair.create(new Mean().evaluate(ap), cmcArray);
    }
}
