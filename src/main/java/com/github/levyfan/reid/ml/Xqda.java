package com.github.levyfan.reid.ml;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.feature.Feature;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;

import java.util.List;

/**
 * @author fanliwen
 */
public class Xqda extends KissMe {

    @Override
    public RealMatrix apply(List<BowImage> bowImages, Feature.Type type) {
        Pair<RealMatrix, RealMatrix> pair = pairPositiveNegative(bowImages, type);
        RealMatrix intra = pair.getFirst();
        RealMatrix extra = pair.getSecond();

        EigenDecomposition ed = new EigenDecomposition(inv(intra).multiply(extra));
        RealMatrix V = ed.getV();

        RealMatrix W;
        switch (type) {
            case HSV:
                W = V.getSubMatrix(0, V.getRowDimension()-1, 0, 49);
                break;
            case SILTP:
                W = V.getSubMatrix(0, V.getRowDimension()-1, 0, 99);
                break;
            case HOG:
            case CN:
            default:
                W = V;
                break;
        }

        RealMatrix positive = W.transpose().multiply(intra).multiply(W);
        RealMatrix negative = W.transpose().multiply(extra).multiply(W);

        RealMatrix M = W.multiply(inv(positive).subtract(inv(negative))).multiply(W.transpose());
        return validateCovMatrix(M);
    }
}
