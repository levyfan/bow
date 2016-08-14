package com.github.levyfan.reid.bow;

import com.github.levyfan.reid.App;
import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.feature.FeatureManager;
import com.google.common.primitives.Doubles;

/**
 * @author fanliwen
 */
public class BowManager {

    private Bow bow;
    private FeatureManager featureManager;
    private boolean wordLevel;

    public BowManager(Bow bow, FeatureManager featureManager, boolean wordLevel) {
        this.bow = bow;
        this.featureManager = featureManager;
        this.wordLevel = wordLevel;
    }

    public Bow getBow() {
        return bow;
    }

    public void bow(BowImage bowImage) {
        featureManager.feature(bowImage);
        
        for (Feature.Type type : App.types) {
            bow.bow(bowImage, type, 0);
//            double[] hist_0 = bowImage.hist.get(type);
//
//            bow.bow(bowImage, type, 1);
//            double[] hist_1 = bowImage.hist.get(type);
//
//            bow.bow(bowImage, type, 2);
//            double[] hist_2 = bowImage.hist.get(type);
//
//            bow.bow(bowImage, type, 3);
//            double[] hist_3 = bowImage.hist.get(type);
//
//            // fusion
//            double[] hist = Doubles.concat(hist_0, hist_1, hist_2, hist_3);
//            bowImage.hist.put(type, hist);
        }

        if (wordLevel) {
            bow.bow(bowImage, Feature.Type.ALL, 0);
//            double[] hist_0 = bowImage.hist.get(Feature.Type.ALL);
//
//            bow.bow(bowImage, Feature.Type.ALL, 1);
//            double[] hist_1 = bowImage.hist.get(Feature.Type.ALL);
//
//            bow.bow(bowImage, Feature.Type.ALL, 2);
//            double[] hist_2 = bowImage.hist.get(Feature.Type.ALL);
//
//            bow.bow(bowImage, Feature.Type.ALL, 3);
//            double[] hist_3 = bowImage.hist.get(Feature.Type.ALL);
//
//            // fusion
//            double[] hist = Doubles.concat(hist_0, hist_1, hist_2, hist_3);
//            bowImage.hist.put(Feature.Type.ALL, hist);
        }

        bowImage.sp4 = null;
        bowImage.strip4 = null;
    }

    @Override
    public String toString() {
        return "BowManager{" +
                "bow=" + bow +
                ", featureManager=" + featureManager +
                ", wordLevel=" + wordLevel +
                '}';
    }
}
