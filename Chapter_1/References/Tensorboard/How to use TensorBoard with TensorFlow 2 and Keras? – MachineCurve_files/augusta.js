window.ezoChar=function(slot,elm){var h=elm.style.minHeight.slice(0,-2);var w=elm.style.minWidth.slice(0,-2);var size=window.ezoCharSize(h,w);if(size=='0'){return false;}
var url='https://go.ezodn.com/charity/https/charity-ads.s3.amazonaws.com/charity_ads/'+size+'.png';var click_url='https://www.cormghana.org/sponsor?utm_source=ezoic';if(size=='300x600'||size=='728x90'){click_url='https://www.redcrossblood.org/donate-blood/dlp/plasma-donations-from-recovered-covid-19-patients.html?utm_source=ezoic';}
var div=document.createElement('div');div.id=elm.id+'_charity';innerElm=elm.childNodes[0];if(innerElm.tagName.toLowerCase()=='script'){innerElm=elm.childNodes[1];}
if(slot.isEmpty==false||innerElm.innerHTML!==""){return true}
var a=document.createElement('a');a.href=click_url;var img=document.createElement('img');img.src=url;a.appendChild(img);div.appendChild(a);innerElm.appendChild(div);__ez.pel.Add(slot,[(new __ezDotData("stat_source_id",11303))],0,0,0,slot.Targeting.br1[0],11303);return true;};window.ezoCharSize=function(h,w){if(w>=728){if(h>=600){return '300x600';}
return '728x90';}
if(w>=300){if(h>=600){return '300x600';}else if(h>=250){return '300x250';}else if(h>=50){return '320x50';}
return '0';}
if(w>=234){if(h>=60){return '234x60';}
return '0';}
if(w>=160){if(h>=90){return '160x90';}
return '0';}
if(w>=100){if(h>=480){return '100x480';}else if(h>=240){return '100x240';}
return '0';}
return '0';};