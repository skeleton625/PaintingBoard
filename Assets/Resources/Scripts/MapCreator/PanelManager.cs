using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PanelManager : MonoBehaviour
{
    [SerializeField]
    private Transform theCamera;
    [SerializeField]
    private GameObject mainButton;
    [SerializeField]
    private GameObject movedPos;
    private Vector3 originPos;
    private Vector3 applyPos;
    [SerializeField]
    private float createPosX;
    [SerializeField]
    private float removePosX;

    private bool activateMainBar;

    /* 기존 카메라 위치 초기화 */
    void Start()
    {
        originPos = theCamera.localPosition;
    }

    void Update()
    {
        /* 특정 영역에서 메인 버튼 창을 표시 */
        ActivateMainBar();
    }

    /* 메인 버튼 창을 표시해주는 함수 */
    private void ActivateMainBar()
    {
        if (!activateMainBar && Input.mousePosition.x <= createPosX)
        {
            MapManager.isPanelOn = true;
            activateMainBar = true;
            mainButton.SetActive(activateMainBar);
            applyPos = movedPos.transform.localPosition;
            StopAllCoroutines();
            StartCoroutine(cameraMovingCoroutine());
        } else if(activateMainBar && Input.mousePosition.x > removePosX)
        {
            MapManager.isPanelOn = false;
            activateMainBar = false;
            mainButton.SetActive(activateMainBar);
            applyPos = originPos;
            StopAllCoroutines();
            StartCoroutine(cameraMovingCoroutine());
        }
    }


    /* 부드럽게 카메라가 이동하도록 하는 코루틴 함수 */
    private IEnumerator cameraMovingCoroutine()
    {
        Vector3 _posV = theCamera.localPosition;
        while(_posV != applyPos)
        {
            _posV = Vector3.Lerp(_posV, applyPos, 0.3f);
            theCamera.localPosition = _posV;
            yield return null;
        }
        theCamera.localPosition = applyPos;
    }
}
