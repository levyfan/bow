package com.github.levyfan.reid.bow;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.feature.FeatureManager;

/**
 * @author fanliwen
 */
public class BowManager {

    private Bow bow;
    private FeatureManager featureManager;

    public BowManager(Bow bow, FeatureManager featureManager) {
        this.bow = bow;
        this.featureManager = featureManager;
    }

    public void bow(BowImage bowImage) {
        featureManager.feature(bowImage);
        
        for (Feature feature : featureManager.getFeatures()) {
            bow.bow(bowImage, feature.name());
        }
        bow.bow(bowImage, Feature.Type.ALL);
    }
}
