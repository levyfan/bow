package com.github.levyfan.reid.bow;

import com.github.levyfan.reid.App;
import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.feature.FeatureManager;

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

    public void bow(BowImage bowImage) {
        featureManager.feature(bowImage);
        
        for (Feature.Type type : App.types) {
            bow.bow(bowImage, type);
        }

        if (wordLevel) {
            bow.bow(bowImage, Feature.Type.ALL);
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
