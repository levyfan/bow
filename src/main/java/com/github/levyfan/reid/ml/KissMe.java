package com.github.levyfan.reid.ml;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.bow.Strip;
import com.github.levyfan.reid.feature.Feature;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;

import java.util.List;
import java.util.Objects;

/**
 * @author fanliwen
 */
public class KissMe {

    public RealMatrix kissMe(List<BowImage> bowImages, Feature.Type type) {
        int length = bowImages.get(0).sp4[0].features.get(type).length;
        RealMatrix positive = MatrixUtils.createRealMatrix(length, length);
        RealMatrix negative = MatrixUtils.createRealMatrix(length, length);

        long countPositive = 0;
        long countNegative = 0;

        for (int i = 0; i < bowImages.size(); i++) {
            for (int j = i + 1; j < bowImages.size(); j++) {
                BowImage x = bowImages.get(i);
                BowImage y = bowImages.get(j);
                System.out.println("kissme: " + x.id);

                Pair<RealMatrix, Long> pair = km(x, y, type);

                if (Objects.equals(x.id, y.id)) {
                    // positive
                    positive = positive.add(pair.getFirst());
                    countPositive += pair.getSecond();
                } else {
                    // negative
                    negative = negative.add(pair.getFirst());
                    countNegative += pair.getSecond();
                }
            }
        }

        positive = positive.scalarMultiply(1.0/(double) countPositive);
        negative = negative.scalarMultiply(1.0/(double) countNegative);

        return new LUDecomposition(positive.subtract(negative)).getSolver().getInverse();
    }

    private static Pair<RealMatrix, Long> km(BowImage i, BowImage j, Feature.Type type) {
        int length = i.sp4[0].features.get(type).length;
        RealMatrix matrix = MatrixUtils.createRealMatrix(length, length);

        long count = 0;

        for (int nstrip = 0; nstrip < i.strip4.length; nstrip ++) {
            Strip iStrip = i.strip4[nstrip];
            Strip jStrip = j.strip4[nstrip];

            for (int iSuperPixel : iStrip.superPixels) {
                for (int jSuperPixel : jStrip.superPixels) {
                    ArrayRealVector iFeature = new ArrayRealVector(i.sp4[iSuperPixel].features.get(type), false);
                    ArrayRealVector jFeature = new ArrayRealVector(j.sp4[jSuperPixel].features.get(type), false);

                    ArrayRealVector delta = iFeature.subtract(jFeature);
                    matrix = matrix.add(delta.outerProduct(delta));

                    count++;
                }
            }
        }
        return Pair.create(matrix, count);
    }
}
