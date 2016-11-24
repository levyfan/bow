package com.github.levyfan.reid.ml;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.feature.Feature;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;

import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * @author fanliwen
 */
public class Xqda extends KissMe {

    @Override
    public RealMatrix apply(List<BowImage> bowImages, Feature.Type type) {
        Pair<RealMatrix, RealMatrix> pair = pairPositiveNegative(bowImages, type);
        RealMatrix intra = pair.getFirst();
        RealMatrix extra = pair.getSecond();

        for (int i = 0; i < intra.getRowDimension(); i ++) {
            intra.setEntry(i, i, intra.getEntry(i, i) + 0.001);
        }

        EigenDecomposition ed = new EigenDecomposition(inv(intra).multiply(extra));
        double[] eigen = ed.getRealEigenvalues();

        int[] sortedIndex = IntStream.range(0, eigen.length).boxed()
                .sorted((i, j) -> Double.compare(eigen[j], eigen[i]))
                .mapToInt(Integer::intValue)
                .toArray();

        int[] selectedRows = IntStream.range(0, eigen.length).toArray();
        int[] selectedColumns;
        switch (type) {
            case HSV:
                selectedColumns = Arrays.copyOf(sortedIndex, 50);
                break;
            case SILTP:
                selectedColumns = Arrays.copyOf(sortedIndex, 100);
                break;
            case HOG:
            case CN:
            default:
                selectedColumns = sortedIndex;
                break;
        }
        RealMatrix W = ed.getV().getSubMatrix(selectedRows, selectedColumns);

        RealMatrix positive = W.transpose().multiply(intra).multiply(W);
        RealMatrix negative = W.transpose().multiply(extra).multiply(W);

        RealMatrix M = inv(positive).subtract(inv(negative));
        RealMatrix MM = validateCovMatrix(M);
        return W.multiply(MM).multiply(W.transpose());
    }

    @Override
    public RealMatrix apply(List<BowImage> bowImages) {
        Pair<RealMatrix, RealMatrix> pair = pairPositiveNegative(bowImages);
        RealMatrix intra = pair.getFirst();
        RealMatrix extra = pair.getSecond();

        for (int i = 0; i < intra.getRowDimension(); i ++) {
            intra.setEntry(i, i, intra.getEntry(i, i) + 0.001);
        }

        EigenDecomposition ed = new EigenDecomposition(inv(intra).multiply(extra));
        double[] eigen = ed.getRealEigenvalues();

        int r = (int) DoubleStream.of(eigen).filter(d -> d > 1).count();
        int[] sortedIndex = IntStream.range(0, eigen.length).boxed()
                .sorted((i, j) -> Double.compare(eigen[j], eigen[i]))
                .mapToInt(Integer::intValue)
                .toArray();

        int[] selectedRows = IntStream.range(0, eigen.length).toArray();
        int[] selectedColumns = Arrays.copyOf(sortedIndex, r);

        RealMatrix W = ed.getV().getSubMatrix(selectedRows, selectedColumns);

        RealMatrix positive = W.transpose().multiply(intra).multiply(W);
        RealMatrix negative = W.transpose().multiply(extra).multiply(W);

        RealMatrix M = inv(positive).subtract(inv(negative));
        // RealMatrix MM = validateCovMatrix(M);
        return W.multiply(M).multiply(W.transpose());
    }
}
